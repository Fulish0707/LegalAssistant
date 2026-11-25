"""Pure metrics and evidence preparation for recall experiments."""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean

DOMAINS = ("criminal", "civil", "administrative", "ip")
SEED = 20260928
# Matches Flask's legal evidence separator; parity is checked before live reporting.
SEPARATOR = "\n…\n"


def validated_review_grades(blank_rows, reviewed_rows):
    """Accept only complete reviewer grades on the exact frozen calibration pairs."""
    blank={row["review_id"]:row for row in blank_rows}
    reviewed={row["review_id"]:row for row in reviewed_rows}
    if (len(blank)!=len(blank_rows) or len(reviewed)!=len(reviewed_rows) or
        blank.keys()!=reviewed.keys()):
        raise ValueError("review membership differs or contains duplicates")
    grades={}
    for review_id,source in blank.items():
        current=reviewed[review_id]
        for key,value in source.items():
            if key not in ("grade","note") and current.get(key)!=value:
                raise ValueError(review_id+": content changed")
        value=(current.get("grade") or "").strip()
        if value not in ("0","1","2"):
            raise ValueError(review_id+": missing or invalid grade")
        pair=(source["case_id"],source["chunk_id"])
        if pair in grades:
            raise ValueError("duplicate case/chunk pair")
        grades[pair]=int(value)
    return grades

def sample_review_calibration(pool, provenance, domain_by_case, *, seed=SEED):
    """Draw 25 blind pairs per domain before any relevance grade is available.

    Source membership and retrieval origin are sampling proxies only; they are not
    relevance labels. The reviewer output shows the same 3,500-character prefix
    used by the existing local relevance judge.
    """
    origins={row["review_id"]:row for row in provenance}
    if len(origins)!=len(provenance) or len({r["review_id"] for r in pool})!=len(pool):
        raise ValueError("duplicate review IDs")
    buckets=defaultdict(list)
    for row in pool:
        domain=domain_by_case[row["case_id"]]
        source=origins[row["review_id"]]["origins"]
        stratum=("reference" if "reference_document" in source else
                 "baseline" if "baseline_top20" in source else "variant")
        buckets[(domain,stratum)].append(row)
    rng=random.Random(seed)
    chosen=[]
    hidden=[]
    for domain in DOMAINS:
        for stratum,quota in (("reference",8),("baseline",9),("variant",8)):
            candidates=sorted(buckets[(domain,stratum)],key=lambda row:row["review_id"])
            if len(candidates)<quota:
                raise ValueError(f"{domain}/{stratum}: need {quota}, have {len(candidates)}")
            for row in rng.sample(candidates,quota):
                chosen.append({key:row[key] for key in
                               ("review_id","case_id","query","chunk_id","document_id","document_type")} |
                              {"text":row["text"][:3500],"grade":"","note":""})
                hidden.append({"review_id":row["review_id"],"domain":domain,"stratum":stratum})
    order=sorted(range(len(chosen)),key=lambda i:chosen[i]["review_id"])
    return [chosen[i] for i in order],[hidden[i] for i in order]

def normalize_query(query):
    return " ".join(unicodedata.normalize("NFC", query).split())


def reference_metrics(hits, references, k=10):
    references = set(references)
    docs = list(dict.fromkeys(h["document_id"] for h in hits))
    first = {h["document_id"] for h in hits[:k]}
    def recall(found):
        return len(set(found) & references) / len(references) if references else None
    return {"reference_coverage": recall(docs), "chunk_cutoff_recall": recall(first),
            "unique_document_recall": recall(docs[:k]), "returned_chunks":len(hits),
            "returned_documents":len(docs),
            "first_reference_chunk_rank":next((i for i,h in enumerate(hits,1) if h["document_id"] in references),None)}


def select_document_aware(hits, *, k=10, max_documents=5):
    """R3: first chunk of five documents, then other chunks from those documents."""
    if k<0 or max_documents<0:
        raise ValueError("negative selection limit")
    if k==0 or max_documents==0:
        return []
    first=[]
    chosen_documents=set()
    chosen_chunks=set()
    for hit in hits:
        if hit["document_id"] not in chosen_documents:
            first.append(hit)
            chosen_documents.add(hit["document_id"])
            chosen_chunks.add(hit["chunk_id"])
            if len(first)==min(k,max_documents): break
    for hit in hits:
        if len(first)>=k: break
        if hit["document_id"] in chosen_documents and hit["chunk_id"] not in chosen_chunks:
            first.append(hit)
            chosen_chunks.add(hit["chunk_id"])
    return first

def prepare_evidence(hits, max_documents=5, char_limit=1200, separator=SEPARATOR):
    documents = {}
    for hit in hits:
        text = hit["chunk"]["text"].strip()
        if not text:
            continue
        doc = hit["document_id"]
        if doc not in documents:
            if len(documents) >= max_documents:
                continue
            documents[doc] = {"document_id":doc,"texts":[],"segments":[]}
        entry = documents[doc]
        if text in entry["texts"]:
            continue
        prefix = len(separator.join(entry["texts"])) + (len(separator) if entry["texts"] else 0)
        offset = len(hit["chunk"]["text"]) - len(hit["chunk"]["text"].lstrip())
        retained = min(len(text),max(0,char_limit-prefix))
        entry["texts"].append(text)
        if retained:
            entry["segments"].append({"chunk_id":hit["chunk_id"],"source_start":offset,
                                      "source_end":offset+retained,"evidence_start":prefix,
                                      "evidence_end":prefix+retained})
    result=[]
    for entry in documents.values():
        joined=separator.join(entry.pop("texts"))
        result.append({**entry,"text":joined[:char_limit],"truncated":len(joined)>char_limit,
                       "untruncated_characters":len(joined)})
    return result


def support_survives(evidence, spans):
    return any(segment["chunk_id"]==span["chunk_id"]
               and segment["source_start"]<=span["start"]<span["end"]<=segment["source_end"]
               for doc in evidence for segment in doc["segments"] for span in spans)


def judged_metrics(hits, judgments, k=10, threshold=1):
    grades=[]
    for hit in hits[:k]:
        grade=judgments.get(hit["chunk_id"])
        if grade not in (0,1,2):
            raise ValueError("missing or failed judgment for "+hit["chunk_id"])
        grades.append(grade)
    return {"success":int(any(g>=threshold for g in grades)),
            "precision":sum(g>=threshold for g in grades)/k}


def paired_interval(baseline, variant, resamples=10000, seed=SEED):
    left={r["case_id"]:r for r in baseline}
    right={r["case_id"]:r for r in variant}
    if len(left)!=len(baseline) or len(right)!=len(variant) or left.keys()!=right.keys() or not left:
        raise ValueError("case membership must match and be nonempty/unique")
    groups=defaultdict(lambda:defaultdict(list))
    for case,a in left.items():
        b=right[case]
        if (a["domain"],a["group_id"])!=(b["domain"],b["group_id"]):
            raise ValueError("case grouping differs")
        groups[a["domain"]][a["group_id"]].append(b["value"]-a["value"])
    # Reject cross-domain clusters: independent stratified resampling would break them.
    domains_by_group=defaultdict(set)
    for domain,g in groups.items():
        for key in g:
            domains_by_group[key].add(domain)
    if any(len(v)>1 for v in domains_by_group.values()):
        raise ValueError("cross-domain groups require joint stratified bootstrap")
    by_domain={d:list(g.values()) for d,g in groups.items()}
    point=mean(mean(x for g in gs for x in g) for gs in by_domain.values())
    rng=random.Random(seed)
    boot=[]
    for _ in range(resamples):
        boot.append(mean(mean(x for g in rng.choices(gs,k=len(gs)) for x in g) for gs in by_domain.values()))
    boot.sort()
    return {"delta":point,"ci95":[boot[int(.025*(resamples-1))],boot[int(.975*(resamples-1))]],
            "resamples":resamples,"seed":seed,"group_count":sum(len(gs) for gs in by_domain.values())}


def development_target(eval_dir:Path, seed=SEED):
    rng=random.Random(seed)
    seen=set()
    selected=[]
    for dataset in [*(d+"-pilot-200" for d in DOMAINS),"everyday-150"]:
        rows=[json.loads(l) for l in (eval_dir/(dataset+".jsonl")).read_text().splitlines() if l.strip()]
        eligible=[]
        for row in sorted(rows,key=lambda r:str(r["id"])):
            query=normalize_query(row["query"])
            if query in seen:
                continue
            seen.add(query)
            domain=dataset.split("-")[0] if dataset!="everyday-150" else row.get("dataset","unknown").split("-")[0]
            eligible.append({**row,"case_id":dataset+":"+str(row["id"]),"dataset":dataset,"domain":domain})
        if len(eligible)<20:
            raise ValueError("fewer than 20 unique questions in "+dataset)
        selected.extend(rng.sample(eligible,20))
    # Connected groups for shared source documents or normalized formal/paraphrased questions.
    parents={r["case_id"]:r["case_id"] for r in selected}
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]]
            x=parents[x]
        return x
    owners={}
    for r in selected:
        keys=["doc:"+d for d in r.get("relevant_document_ids",[])]
        keys += ["q:"+normalize_query(r[k]) for k in ("query","formal_query") if r.get(k)]
        for key in keys:
            if key in owners:
                a,b=find(r["case_id"]),find(owners[key])
                parents[max(a,b)]=min(a,b)
            else:
                owners[key]=r["case_id"]
    for r in selected:
        r["group_id"]=find(r["case_id"])
    return selected


def dataset_audit(eval_dir:Path):
    inventory={}
    owners=defaultdict(list)
    for name in [*(d+"-pilot-200" for d in DOMAINS),"everyday-150","targets/crossdomain-60/target"]:
        p=eval_dir/(name+".jsonl")
        if not p.exists():
            inventory[name]={"missing":True}
            continue
        raw=p.read_bytes()
        rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
        counts=defaultdict(int)
        for r in rows:
            q=normalize_query(r["query"])
            counts[q]+=1
            owners[q].append(name+":"+str(r["id"]))
        inventory[name]={"sha256":hashlib.sha256(raw).hexdigest(),"rows":len(rows),
                         "duplicate_queries":sum(n-1 for n in counts.values()),
                         "missing_reference_rows":sum(not r.get("relevant_document_ids") for r in rows)}
    return {"files":inventory,"duplicate_query_groups":[v for v in owners.values() if len(v)>1]}


def summarize(rows):
    by_domain=defaultdict(lambda:defaultdict(list))
    failures=0
    for row in rows:
        if row.get("errors"):
            failures+=1
            continue
        for stage, metrics in row.get("reference_metrics",{}).items():
            for key,value in metrics.items():
                if isinstance(value,(int,float)):
                    by_domain[row["domain"]][stage+"."+key].append(value)
    domains={d:{k:mean(v) for k,v in metrics.items()} for d,metrics in by_domain.items()}
    keys={k for values in domains.values() for k in values}
    macro={k:mean(values[k] for values in domains.values() if k in values) for k in keys}
    return {"requests":len(rows),"failed_requests":failures,
            "quality_status":"unjudged_known_reference_only",
            "domains":domains,"macro":macro,
            "limitations":["Known references are incomplete relevance labels.",
                           "No semantic-recall or adoption claim without independent judgments.",
                           "Stage means condition on successful requests; failures remain counted separately."]}
