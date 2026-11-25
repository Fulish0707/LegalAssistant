#!/usr/bin/env python
"""Local stage-aware recall experiments. Every output directory is new."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from legal_rag.evaluation.recall_report import (
    dataset_audit, development_target, prepare_evidence, reference_metrics,
    sample_review_calibration, select_document_aware, summarize,
    validated_review_grades)
from legal_rag.evaluation.recall_stages import capture_stages, capture_multi_stages, replay_stages
from legal_rag.evaluation.recall_variants import VARIANTS

def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")

def append(handle, value):
    handle.write(json.dumps(value,ensure_ascii=False)+"\n")
    handle.flush()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(args):
    result=subprocess.run(["git",*args],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None

def manifest(args, settings=None):
    out={"started_at":datetime.now(timezone.utc).isoformat(),"command":vars(args),
         "git_head":git(["rev-parse","HEAD"]),"git_status":git(["status","--short"]),
         "python":sys.version,"packages":{},
         "status":"running","evaluation":"development_only"}
    out["command"]={k:str(v) if isinstance(v,Path) else v for k,v in out["command"].items()}
    for package in ("qdrant-client","httpx","torch","sentence-transformers","pydantic"):
        try: out["packages"][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    source=Path(__file__).resolve().parents[1]
    out["source_hashes"]={str(p.relative_to(source)):sha(p)
                         for p in sorted((source/"src/legal_rag/evaluation").glob("recall_*.py"))}
    out["source_hashes"]["scripts/eval_recall_stages.py"]=sha(Path(__file__))
    if settings:
        out["settings"]=settings.model_dump(mode="json",exclude={"qdrant":{"api_key"}})
        out["index_metadata"]={}
        for corpus in settings.retrieval.corpora:
            p=args.data_root/"data/indexes"/corpus/"index.json"
            out["index_metadata"][corpus]={"sha256":sha(p),"metadata":json.loads(p.read_text())}
        import httpx
        out["ollama_models"]=httpx.get(settings.embedding.base_url+"/api/tags",timeout=10).json()["models"]
        out["gpu"]=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.used",
                                   "--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
    return out

def errors(record):
    return [s["error"] for s in [*record.get("legs",[]),record["fused"],record["reranked"]]
            if s.get("error")] + record.get("input_errors",[])

def compact(record):
    """Variants reuse capture payloads; retain IDs and scores without repeating text."""
    return {key:value for key,value in record.items() if key not in ("legs","fused","reranked")} | {
        name:{**record[name],"hits":[{k:v for k,v in h.items() if k!="chunk"} for h in record[name]["hits"]]}
        for name in ("fused","reranked")}

def metrics(case, record, union):
    refs=case.get("relevant_document_ids",[])
    top=record["reranked"]["hits"][:10]
    evidence=prepare_evidence(top)
    retained=[h for h in top if any(h["chunk_id"]==s["chunk_id"] for e in evidence for s in e["segments"])]
    union_metrics=reference_metrics(union,refs)
    union_metrics={key:union_metrics[key] for key in
                   ("reference_coverage","returned_chunks","returned_documents")}
    return {"case_id":case["case_id"],"domain":case["domain"],"group_id":case["group_id"],
            "reference_metrics":{"union":union_metrics,
                "fused":reference_metrics(record["fused"]["hits"],refs),
                "top10":reference_metrics(top,refs),"prepared":reference_metrics(retained,refs)},
            "errors":errors(record),"evidence":evidence}

class CachedReranker:
    """Cache is per query and bound to the exact input text and one model instance."""
    def __init__(self,delegate):
        self.delegate=delegate
        self.scores={}
    def key(self,q,c):
        return (q,c.id,hashlib.sha256(c.text.encode()).hexdigest())
    def rerank(self,query,candidates,k):
        missing=[c for c in candidates if self.key(query,c.chunk) not in self.scores]
        if missing:
            ranked=self.delegate.rerank(query,missing,len(missing))
            if len(ranked)!=len(missing):
                raise ValueError("reranker did not score every candidate")
            for hit in ranked:
                self.scores[self.key(query,hit.chunk)]=hit.score
        indexed=list(enumerate(candidates))
        indexed.sort(key=lambda item:(-self.scores[self.key(query,item[1].chunk)],item[0]))
        return [h.model_copy(update={"score":self.scores[self.key(query,h.chunk)],"retriever":"reranked"})
                for _,h in indexed[:k]]

def audit(args):
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"manifest.json",manifest(args))
    write_json(args.output/"dataset-audit.json",dataset_audit(args.data_root/"data/eval"))
    target=development_target(args.data_root/"data/eval")
    with (args.output/"target.jsonl").open("x") as f:
        for row in target: append(f,row)
    write_json(args.output/"status.json",{"status":"complete","questions":len(target),
        "target_sha256":sha(args.output/"target.jsonl"),"confirmation":"pending_fresh_questions_and_independent_labels"})
    print(json.dumps({"questions":len(target),"output":str(args.output)}))

def capture(args):
    from legal_rag.config import load_settings
    from legal_rag.factory import build_hybrid_retriever, close_legs
    settings=load_settings(profile="local",overrides={"paths":{"root":str(args.data_root)}})
    # Explicit fixed baseline is required for meaningful named variant replays.
    actual=(settings.retrieval.k_dense,settings.retrieval.k_sparse,settings.retrieval.k_fused,
            settings.retrieval.rrf_k,settings.retrieval.dense_weight,settings.retrieval.sparse_weight,
            settings.retrieval.leg_floor,settings.rerank.dtype)
    if actual!=(100,100,100,60,1,.75,3,"float16"):
        raise ValueError("resolved baseline differs from plan: "+repr(actual))
    if settings.retrieval.corpus_weights:
        raise ValueError("nondefault corpus weights cannot use fixed variants")
    cases=[json.loads(l) for l in args.target.read_text().splitlines() if l.strip()]
    if args.limit: cases=cases[:args.limit]
    if not cases: raise ValueError("empty target")
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=manifest(args,settings)
    provenance["target_sha256"]=sha(args.target)
    write_json(args.output/"manifest.json",provenance)
    with (args.output/"target.jsonl").open("x") as f:
        for case in cases: append(f,case)
    try:
        retriever=build_hybrid_retriever(settings)
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        write_json(args.output/"manifest.json",provenance)
        raise
    real= retriever.reranker
    aggregates={v:[] for v in args.variants}
    parity=[]
    parity_ids=set()
    full_union_ids=set()
    domains=sorted({c["domain"] for c in cases})
    queues=[[c for c in cases if c["domain"]==d] for d in domains]
    interleaved=[c for n in range(max(map(len,queues))) for q in queues for c in q[n:n+1]]
    parity_ids={c["case_id"] for c in interleaved[:args.parity]}
    full_union_ids={c["case_id"] for q in queues for c in q[:10]}
    if args.full_union: aggregates["full_union"]=[]
    started=time.perf_counter()
    try:
        for i in range(args.warmup):
            retriever.retrieve(cases[i%len(cases)]["query"],10,None)
        with (args.output/"stages.jsonl").open("x") as captures, \
             (args.output/"variants.jsonl").open("x") as variants, \
             (args.output/"per-case.jsonl").open("x") as percase:
            for i,case in enumerate(cases):
                t=time.perf_counter()
                cached=CachedReranker(real)
                retriever.reranker=cached
                rec=capture_stages(retriever,case["query"],dense_k=args.depth,sparse_k=args.depth,rerank_k=100)
                if errors(rec):
                    append(captures,{"case":case,"capture":rec})
                    raise RuntimeError("failed retrieval stage: "+repr(errors(rec)))
                if case["case_id"] in parity_ids:
                    retriever.reranker=real
                    live=retriever.retrieve(case["query"],10,None)
                    match=[h.chunk.id for h in live]==[h["chunk_id"] for h in rec["reranked"]["hits"][:10]]
                    parity.append({"case_id":case["case_id"],"match":match})
                    write_json(args.output/"parity.json",parity)
                    if not match:
                        raise RuntimeError("baseline parity mismatch; stop before comparisons")
                # All variants can reuse exact pointwise scores. Union is a diagnostic only.
                union=list({h["chunk_id"]:h for leg in rec["legs"] for h in leg["hits"]}.values())
                if args.full_union and case["case_id"] in full_union_ids:
                    full=replay_stages(rec,cached,full_union=True,rerank_k=len(union))
                    if errors(full): raise RuntimeError(repr(errors(full)))
                    append(variants,{"case_id":case["case_id"],"variant":"full_union","record":compact(full)})
                    full_result=metrics(case,full,union)
                    aggregates["full_union"].append(full_result)
                    append(percase,{"variant":"full_union",**full_result})
                for name in args.variants:
                    replay=rec if name=="baseline" else replay_stages(rec,cached,variant=name,rerank_k=100)
                    if errors(replay): raise RuntimeError(repr(errors(replay)))
                    result=metrics(case,replay,union)
                    aggregates[name].append(result)
                    append(variants,{"case_id":case["case_id"],"variant":name,"record":compact(replay)})
                    append(percase,{"variant":name,**result})
                append(captures,{"case":case,"capture":rec,
                    "scores":[{"query":q,"chunk_id":c,"text_sha256":h,"score":s} for (q,c,h),s in cached.scores.items()]})
                print(json.dumps({"completed":i+1,"total":len(cases),"case_id":case["case_id"],
                                  "union":len(union),"seconds":round(time.perf_counter()-t,3)}),flush=True)
        write_json(args.output/"summary.json",{name:summarize(rows) for name,rows in aggregates.items()})
        provenance.update(status="complete",elapsed_seconds=time.perf_counter()-started,parity_queries=len(parity),
                          judged_quality="not_evaluated",confirmation="not_performed")
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        write_json(args.output/"manifest.json",provenance)
        retriever.reranker=real
        close_legs(retriever)


def ann_comparison(approximate, exact, relevant_document_ids):
    approximate_ids={h["chunk_id"] for h in approximate}
    exact_ids={h["chunk_id"] for h in exact}
    approximate_documents={h["document_id"] for h in approximate}
    exact_documents={h["document_id"] for h in exact}
    rescued=(exact_documents-approximate_documents) & set(relevant_document_ids)
    return {"overlap":len(approximate_ids & exact_ids)/len(exact_ids) if exact_ids else None,
            "rescued_reference_ids":sorted(rescued)}

def ann_search_modes(efs):
    from qdrant_client import models
    return [("approximate",None),
            *[(f"ef{ef}",models.SearchParams(hnsw_ef=ef)) for ef in efs],
            ("exact",models.SearchParams(exact=True))]

def ann(args):
    from collections import defaultdict
    from legal_rag.config import load_settings
    from legal_rag.factory import build_hybrid_retriever, close_legs
    settings=load_settings(profile="local",overrides={"paths":{"root":str(args.data_root)}})
    rows=[json.loads(l) for l in args.target.read_text().splitlines() if l.strip()]
    selected=[]
    for domain in ("criminal","civil","administrative","ip"):
        selected.extend([r for r in rows if r["domain"]==domain][:10])
    if len(selected)!=40:
        raise ValueError("ANN diagnostic requires ten cases per domain")
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=manifest(args,settings)
    provenance["target_sha256"]=sha(args.target)
    write_json(args.output/"manifest.json",provenance)
    try:
        retriever=build_hybrid_retriever(settings)
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        write_json(args.output/"manifest.json",provenance)
        raise
    results=[]
    try:
        dense=[leg for leg in retriever.legs if leg.name.endswith(":dense")]
        with (args.output/"ann.jsonl").open("x") as output:
            for i,case in enumerate(selected):
                vector=dense[0].embedder.embed_query(case["query"])
                for leg in dense:
                    store=leg.store
                    rankings={}
                    timing={}
                    for name,params in ann_search_modes(getattr(args,"efs",[])):
                        started=time.perf_counter()
                        response=store.client.query_points(store.collection,query=vector,using=store.DENSE,
                            limit=100,search_params=params,with_payload=["id","document_id"])
                        timing[name]=(time.perf_counter()-started)*1000
                        rankings[name]=[{"chunk_id":h.payload["id"],"document_id":h.payload["document_id"],
                                         "score":h.score} for h in response.points]
                    comparisons={name:ann_comparison(hits,rankings["exact"],
                                               case.get("relevant_document_ids",[]))
                                 for name,hits in rankings.items() if name!="exact"}
                    comparison=comparisons["approximate"]
                    record={"case_id":case["case_id"],"domain":case["domain"],"leg":leg.name,
                            **comparison,"comparisons":comparisons,"rankings":rankings,"timing_ms":timing}
                    append(output,record)
                    results.append(record)
                print(json.dumps({"ann_completed":i+1,"total":len(selected)}),flush=True)
        by_domain=defaultdict(list)
        for r in results:
            if r["overlap"] is not None: by_domain[r["domain"]].append(r["overlap"])
        effort={}
        for name,_ in ann_search_modes(getattr(args,"efs",[])):
            if name=="exact": continue
            vals=[r["comparisons"][name]["overlap"] for r in results
                  if r["comparisons"][name]["overlap"] is not None]
            effort[name]={"mean_overlap":sum(vals)/len(vals) if vals else None,
                          "reference_rescues":sum(bool(r["comparisons"][name]["rescued_reference_ids"])
                                                   for r in results),
                          "mean_latency_ms":sum(r["timing_ms"][name] for r in results)/len(results)}
        effort["exact"]={"mean_latency_ms":sum(r["timing_ms"]["exact"] for r in results)/len(results)}
        write_json(args.output/"summary.json",{
            "queries":40,"search_pairs":len(results),
            "overlap_by_domain":{d:sum(v)/len(v) for d,v in by_domain.items()},
            "reference_rescues":sum(bool(r["rescued_reference_ids"]) for r in results),
            "search_effort":effort,
            "reviewed_relevance_rescues":"not_judged",
            "limitation":"ANN overlap is not semantic relevance recall."})
        provenance["status"]="complete"
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        write_json(args.output/"manifest.json",provenance)
        close_legs(retriever)


def judging_pool(stages, variants, reference_by_doc):
    cases={row["case"]["case_id"]:row for row in stages}
    pooled={}
    origins={}
    def add(case, chunk, origin, variant=None):
        case_id=case["case_id"]
        chunk_id=chunk["id"]
        review_id=hashlib.sha256((case_id+"\0"+chunk_id).encode()).hexdigest()[:24]
        key=(case_id,chunk_id)
        if key not in pooled:
            pooled[key]={"review_id":review_id,"case_id":case_id,"query":case["query"],
                         "chunk_id":chunk_id,"document_id":chunk["document_id"],
                         "document_type":chunk.get("document_type"),"text":chunk["text"]}
            origins[key]={"review_id":review_id,"origins":set(),"variants":set()}
        origins[key]["origins"].add(origin)
        if variant: origins[key]["variants"].add(variant)
    for case_id,row in cases.items():
        case=row["case"]
        hits={h["chunk_id"]:h["chunk"] for leg in row["capture"]["legs"]
              for h in leg["hits"]}
        for hit in row["capture"]["reranked"]["hits"][:20]:
            add(case,hits[hit["chunk_id"]],"baseline_top20")
        for doc in case.get("relevant_document_ids",[]):
            for chunk in reference_by_doc.get(doc,[]):
                add(case,chunk,"reference_document")
    for row in variants:
        case_id=row["case_id"]
        case=cases[case_id]["case"]
        hits={h["chunk_id"]:h["chunk"] for leg in cases[case_id]["capture"]["legs"]
              for h in leg["hits"]}
        for hit in row["record"]["reranked"]["hits"][:10]:
            add(case,hits[hit["chunk_id"]],"variant_top10",row["variant"])
    ordered=sorted(pooled,key=lambda key:pooled[key]["review_id"])
    return ([pooled[key] for key in ordered],
            [{"review_id":origins[key]["review_id"],
              "origins":sorted(origins[key]["origins"]),
              "variants":sorted(origins[key]["variants"])} for key in ordered])

def extend_judging_pool(pool, provenance, c5_stages, variant_name="C5"):
    by_review={row["review_id"]:row for row in pool}
    origins={row["review_id"]:{**row,"origins":set(row["origins"]),
                               "variants":set(row["variants"])} for row in provenance}
    if len(by_review)!=len(pool) or len(origins)!=len(provenance) or by_review.keys()!=origins.keys():
        raise ValueError("pool/provenance IDs differ or repeat")
    queries={row["case_id"]:row["query"] for row in pool}
    for record in c5_stages:
        case_id=record["case_id"]
        for hit in record["top10"]:
            chunk=hit["chunk"]
            review_id=hashlib.sha256((case_id+"\0"+hit["chunk_id"]).encode()).hexdigest()[:24]
            if review_id not in by_review:
                by_review[review_id]={"review_id":review_id,"case_id":case_id,
                                      "query":queries[case_id],"chunk_id":hit["chunk_id"],
                                      "document_id":hit["document_id"],
                                      "document_type":chunk.get("document_type"),
                                      "text":chunk["text"]}
                origins[review_id]={"review_id":review_id,"origins":set(),"variants":set()}
            else:
                prior=by_review[review_id]
                if prior["case_id"]!=case_id or prior["chunk_id"]!=hit["chunk_id"]:
                    raise ValueError("review ID collision")
            origins[review_id]["origins"].add("variant_top10")
            origins[review_id]["variants"].add(variant_name)
    ids=sorted(by_review)
    return ([by_review[i] for i in ids],
            [{"review_id":i,"origins":sorted(origins[i]["origins"]),
              "variants":sorted(origins[i]["variants"])} for i in ids])

def pool_extend(args):
    base_pool=json.loads((args.pool_dir/"manifest.json").read_text())
    baseline=json.loads((args.baseline/"manifest.json").read_text())
    c5=json.loads((args.c5/"manifest.json").read_text())
    if (base_pool.get("status")!="unjudged" or baseline.get("status")!="complete" or
        c5.get("status")!="complete" or
        base_pool["source_run_manifest_sha256"]!=sha(args.baseline/"manifest.json") or
        baseline["target_sha256"]!=c5["target_sha256"]):
        raise ValueError("source runs or target snapshots do not match")
    pool=[json.loads(line) for line in (args.pool_dir/"review-pool.jsonl").read_text().splitlines() if line]
    provenance=[json.loads(line) for line in (args.pool_dir/"pool-provenance.jsonl").read_text().splitlines() if line]
    c5_stages=[json.loads(line) for line in (args.c5/"stages.jsonl").read_text().splitlines() if line]
    if len(c5_stages)!=base_pool["questions"]:
        raise ValueError("C5 question count differs")
    rows,origins=extend_judging_pool(pool,provenance,c5_stages)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"review-pool.jsonl").open("x") as handle:
        for row in rows: append(handle,row)
    with (args.output/"pool-provenance.jsonl").open("x") as handle:
        for row in origins: append(handle,row)
    write_json(args.output/"manifest.json",{
        "status":"unjudged","questions":base_pool["questions"],"pairs":len(rows),
        "reference_documents":base_pool["reference_documents"],
        "source_run_manifest_sha256":base_pool["source_run_manifest_sha256"],
        "base_pool_manifest_sha256":sha(args.pool_dir/"manifest.json"),
        "c5_manifest_sha256":sha(args.c5/"manifest.json"),
        "c5_stages_sha256":sha(args.c5/"stages.jsonl"),
        "added_pairs":len(rows)-len(pool),
        "instructions":"Review only review-pool.jsonl; variant provenance remains separate."})

def pool_extend_r3(args):
    base_pool=json.loads((args.pool_dir/"manifest.json").read_text())
    baseline=json.loads((args.baseline/"manifest.json").read_text())
    r3=json.loads((args.r3/"manifest.json").read_text())
    if (base_pool.get("status")!="unjudged" or baseline.get("status")!="complete" or
        r3.get("status")!="complete" or
        base_pool["source_run_manifest_sha256"]!=sha(args.baseline/"manifest.json") or
        r3["source_manifest_sha256"]!=sha(args.baseline/"manifest.json")):
        raise ValueError("R3 and pool do not share the frozen baseline")
    pool=[json.loads(line) for line in (args.pool_dir/"review-pool.jsonl").read_text().splitlines() if line]
    provenance=[json.loads(line) for line in (args.pool_dir/"pool-provenance.jsonl").read_text().splitlines() if line]
    stages=[json.loads(line) for line in (args.r3/"top10.jsonl").read_text().splitlines() if line]
    if len(stages)!=base_pool["questions"]:
        raise ValueError("R3 question count differs")
    rows,origins=extend_judging_pool(pool,provenance,stages,variant_name="R3")
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"review-pool.jsonl").open("x") as handle:
        for row in rows: append(handle,row)
    with (args.output/"pool-provenance.jsonl").open("x") as handle:
        for row in origins: append(handle,row)
    write_json(args.output/"manifest.json",{
        "status":"unjudged","questions":base_pool["questions"],"pairs":len(rows),
        "reference_documents":base_pool["reference_documents"],
        "source_run_manifest_sha256":base_pool["source_run_manifest_sha256"],
        "base_pool_manifest_sha256":sha(args.pool_dir/"manifest.json"),
        "r3_manifest_sha256":sha(args.r3/"manifest.json"),
        "r3_top10_sha256":sha(args.r3/"top10.jsonl"),
        "added_pairs":len(rows)-len(pool),
        "instructions":"Review only review-pool.jsonl; variant provenance remains separate."})

def pool_run(args):
    from collections import defaultdict
    manifest_path=args.run_dir/"manifest.json"
    provenance=json.loads(manifest_path.read_text())
    if provenance.get("status")!="complete":
        raise ValueError("judging pool requires a complete run")
    variants_by_case=defaultdict(list)
    with (args.run_dir/"variants.jsonl").open() as source:
        for line in source:
            row=json.loads(line)
            variants_by_case[row["case_id"]].append(row)
    refs={doc for line in (args.run_dir/"target.jsonl").read_text().splitlines()
          if line.strip() for doc in json.loads(line).get("relevant_document_ids",[])}
    reference_by_doc=defaultdict(list)
    for corpus in ("criminal-v1","civil-v1","administrative-v1","ip-v1"):
        with (args.data_root/"data/processed"/corpus/"chunks.jsonl").open() as source:
            for line in source:
                chunk=json.loads(line)
                if chunk["document_id"] in refs:
                    reference_by_doc[chunk["document_id"]].append(chunk)
    if refs-set(reference_by_doc):
        raise ValueError("reference documents missing from processed corpus")
    args.output.mkdir(parents=True,exist_ok=False)
    questions=pairs=0
    with ((args.run_dir/"stages.jsonl").open() as source,
          (args.output/"review-pool.jsonl").open("x") as review,
          (args.output/"pool-provenance.jsonl").open("x") as provenance_out):
        for line in source:
            stage=json.loads(line)
            case_id=stage["case"]["case_id"]
            rows,origins=judging_pool([stage],variants_by_case[case_id],reference_by_doc)
            for row in rows: append(review,row)
            for row in origins: append(provenance_out,row)
            questions+=1
            pairs+=len(rows)
    write_json(args.output/"manifest.json",{
        "status":"unjudged","source_run_manifest_sha256":sha(manifest_path),
        "source_stages_sha256":sha(args.run_dir/"stages.jsonl"),
        "source_variants_sha256":sha(args.run_dir/"variants.jsonl"),
        "questions":questions,"pairs":pairs,"reference_documents":len(refs),
        "instructions":"Review only review-pool.jsonl; pool-provenance.jsonl contains hidden run identity. Assign rubric grade 0, 1, or 2 after independent calibration."})

def r3_report(args):
    from legal_rag.evaluation.recall_report import paired_interval
    source=json.loads((args.baseline/"manifest.json").read_text())
    if source.get("status")!="complete":
        raise ValueError("R3 requires a complete baseline capture")
    baseline={row["case_id"]:row for line in (args.baseline/"per-case.jsonl").read_text().splitlines()
              if line for row in [json.loads(line)] if row["variant"]=="baseline"}
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=manifest(args)
    provenance["source_manifest_sha256"]=sha(args.baseline/"manifest.json")
    provenance["source_stages_sha256"]=sha(args.baseline/"stages.jsonl")
    write_json(args.output/"manifest.json",provenance)
    rows=[]
    try:
        with ((args.baseline/"stages.jsonl").open() as source_rows,
              (args.output/"per-case.jsonl").open("x") as percase,
              (args.output/"top10.jsonl").open("x") as top10):
            for line in source_rows:
                saved=json.loads(line)
                case=saved["case"]
                rec=saved["capture"]
                union=list({h["chunk_id"]:h for leg in rec["legs"] for h in leg["hits"]}.values())
                original=metrics(case,rec,union)
                expected=baseline[case["case_id"]]["reference_metrics"]["top10"]
                if original["reference_metrics"]["top10"]!=expected:
                    raise ValueError("baseline stage/result mismatch")
                selected=select_document_aware(rec["reranked"]["hits"],k=10,max_documents=5)
                shadow={**rec,"reranked":{**rec["reranked"],"hits":selected}}
                result=metrics(case,shadow,union)
                result["variant"]="R3"
                rows.append(result)
                append(percase,result)
                append(top10,{"case_id":case["case_id"],"top10":selected})
        if len(rows)!=len(baseline):
            raise ValueError("R3 question count differs")
        def value(row):
            return {field:row[field] for field in ("case_id","group_id","domain")} | {
                "value":row["reference_metrics"]["top10"]["chunk_cutoff_recall"]}
        comparison=paired_interval([value(baseline[r["case_id"]]) for r in rows],
                                   [value(r) for r in rows])
        result={"decision":"no_adoption_unjudged_development",
                "baseline":summarize(list(baseline.values())),"R3":summarize(rows),
                "paired_top10":comparison,"questions":len(rows)}
        write_json(args.output/"comparison.json",result)
        a=result["baseline"]["macro"]["top10.chunk_cutoff_recall"]
        b=result["R3"]["macro"]["top10.chunk_cutoff_recall"]
        (args.output/"report.md").write_text(
            "# R3 document-aware selection development result\n\n"
            "Known-source diagnostics only; supporting-passage labels are unavailable.\n\n"
            f"Baseline macro source R@10: {a:.3f}; R3: {b:.3f}. "
            f"Paired delta {comparison['delta']:+.3f} "
            f"(exploratory 95% CI [{comparison['ci95'][0]:+.3f}, "
            f"{comparison['ci95'][1]:+.3f}]).\n\n"
            "R3 retains the first chunk from each of the first five documents and then fills "
            "the remaining slots from those documents. It does not establish evidence survival "
            "or answer quality without reviewed supporting passages. No serving change is adopted.\n")
        provenance["status"]="complete"
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        write_json(args.output/"manifest.json",provenance)

def calibration_agreement(args):
    import csv
    from legal_rag.evaluation.calibration import agreement
    from legal_rag.evaluation.judgment_store import JudgmentStore
    from legal_rag.evaluation.relevance_judge import judge_key
    frozen=json.loads((args.calibration/"manifest.json").read_text())
    judged=json.loads((args.judge_run/"manifest.json").read_text())
    if (frozen.get("status")!="awaiting_independent_labels" or
        judged.get("status")!="uncalibrated_complete" or
        judged["calibration_manifest_sha256"]!=sha(args.calibration/"manifest.json") or
        judged["calibration_csv_sha256"]!=sha(args.calibration/"review-calibration.csv")):
        raise ValueError("judge run does not match frozen calibration")
    with (args.calibration/"review-calibration.csv").open(newline="",encoding="utf-8-sig") as handle:
        blank=list(csv.DictReader(handle))
    with args.reviewed_csv.open(newline="",encoding="utf-8-sig") as handle:
        reviewed=list(csv.DictReader(handle))
    reference=validated_review_grades(blank,reviewed)
    if len(reference)!=100:
        raise ValueError("plan requires all 100 calibration grades")
    key=judge_key(judged["judge_model"],think=judged["think"])
    if key.prompt_sha256!=judged["prompt_sha256"]:
        raise ValueError("judge prompt hash differs")
    predicted=JudgmentStore(args.judge_run/"judgments.jsonl").grades(key)
    if set(predicted)!=set(reference):
        raise ValueError("missing, failed, or extra judge grades")
    strata={row["review_id"]:row["domain"] for line in
            (args.calibration/"calibration-strata.jsonl").read_text().splitlines()
            if line for row in [json.loads(line)]}
    domain_by_pair={(row["case_id"],row["chunk_id"]):strata[row["review_id"]]
                    for row in blank}
    overall=agreement(reference,predicted,gold_pairs=set())
    domains={}
    for domain in ("criminal","civil","administrative","ip"):
        subset={pair:grade for pair,grade in reference.items() if domain_by_pair[pair]==domain}
        if len(subset)!=25: raise ValueError("calibration not balanced by domain")
        domains[domain]=agreement(subset,{pair:predicted[pair] for pair in subset},gold_pairs=set())
    kappa=overall["binary_relevant"]["kappa"]
    passed=kappa is not None and kappa>=.60
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"agreement.json",{
        "calibration_gate":"pass" if passed else "fail",
        "predeclared_metric":"Cohen kappa at grade >= 1","threshold":.60,
        "reviewer":args.reviewer,"reviewer_type":args.reviewer_type,
        "reviewed_csv_sha256":sha(args.reviewed_csv),
        "judge_run_manifest_sha256":sha(args.judge_run/"manifest.json"),
        "overall":overall,"per_domain":domains,
        "limitation":("AI-only reviewer labels remain provisional. " if args.reviewer_type=="ai" else "")+
                     "Fresh-question confirmation is separate."})
    if not passed:
        raise ValueError("calibration kappa below predeclared 0.60 gate")

def calibration_judge_pair(row):
    return {"case_id":row["case_id"],"chunk_id":row["chunk_id"],
            "document_id":row["document_id"],"document_type":row["document_type"],
            "question":row["query"],"reference_answer":"","title":"",
            "text":row["text"]}

def judge_calibration(args):
    import csv
    from legal_rag.llm import OllamaChatModel
    from legal_rag.evaluation.judgment_store import JudgmentStore
    from legal_rag.evaluation.relevance_judge import (
        judge_key, judge_pool, generation_settings, ollama_model_digest)
    source=json.loads((args.calibration/"manifest.json").read_text())
    if source.get("status")!="awaiting_independent_labels":
        raise ValueError("expected the frozen unlabelled calibration sample")
    with (args.calibration/"review-calibration.csv").open(newline="",encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=source["pairs"] or any(row["grade"].strip() for row in rows):
        raise ValueError("calibration file must be complete and unlabelled before model judging")
    if args.limit: rows=rows[:args.limit]
    pairs=[calibration_judge_pair(row) for row in rows]
    args.output.mkdir(parents=True,exist_ok=False)
    key=judge_key("ollama:qwen3:8b",think=True)
    provenance={"status":"running","judge_model":key.model,
                "prompt_sha256":key.prompt_sha256,"think":True,
                "generation":generation_settings(think=True),
                "calibration_csv_sha256":sha(args.calibration/"review-calibration.csv"),
                "calibration_manifest_sha256":sha(args.calibration/"manifest.json"),
                "model_digest":ollama_model_digest(args.base_url,"qwen3:8b"),
                "requested_pairs":len(pairs),
                "limitation":"Model grades are uncalibrated until independent labels and kappa gate pass."}
    write_json(args.output/"manifest.json",provenance)
    model=OllamaChatModel(model="qwen3:8b",base_url=args.base_url,timeout=300.0,
                          think=True,num_ctx=generation_settings(think=True)["num_ctx"])
    try:
        counts=judge_pool(pairs,model,JudgmentStore(args.output/"judgments.jsonl"),
                          think=True,progress=lambda message: print(message,flush=True))
        provenance.update(status="uncalibrated_complete" if not args.limit else "uncalibrated_smoke",
                          counts=counts)
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        model._client.close()
        write_json(args.output/"manifest.json",provenance)

def expansion_repeat_ids(cases):
    """Choose the first five frozen development cases from each domain."""
    return {case["case_id"] for domain in ("criminal","civil","administrative","ip")
            for case in [r for r in cases if r["domain"]==domain][:5]}

def expansion_generate(args):
    from collections import Counter
    from legal_rag.config import load_settings
    from legal_rag.llm import OllamaChatModel
    from legal_rag.evaluation.recall_expansion import (
        MODEL, NUM_CTX, generate_expansion, prompt_sha256)
    settings=load_settings(profile="local",overrides={"paths":{"root":str(args.data_root)}})
    cases=[json.loads(line) for line in args.target.read_text().splitlines() if line]
    if args.limit: cases=cases[:args.limit]
    if not cases: raise ValueError("empty target")
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=manifest(args,settings)
    provenance.update(target_sha256=sha(args.target),model=MODEL,
                      prompt_sha256=prompt_sha256(),think=False,num_ctx=NUM_CTX)
    write_json(args.output/"manifest.json",provenance)
    model=OllamaChatModel(model=MODEL,base_url=settings.llm.base_url,
                          timeout=300.0,think=False,num_ctx=NUM_CTX)
    repeats=expansion_repeat_ids(cases)
    counts=Counter()
    stability=[]
    try:
        with (args.output/"expansions.jsonl").open("x") as output:
            for i,case in enumerate(cases,1):
                started=time.perf_counter()
                first=generate_expansion(model,case["query"])
                first["elapsed_ms"]=(time.perf_counter()-started)*1000
                repeated=None
                if case["case_id"] in repeats:
                    started=time.perf_counter()
                    repeated=generate_expansion(model,case["query"])
                    repeated["elapsed_ms"]=(time.perf_counter()-started)*1000
                    stability.append({"case_id":case["case_id"],
                                      "accepted_same":first["legal_query"]==repeated["legal_query"],
                                      "raw_same":first["raw"]==repeated["raw"]})
                counts[first["fallback_reason"] or "accepted"]+=1
                append(output,{"case_id":case["case_id"],"domain":case["domain"],
                               "original_query":case["query"],"first":first,"repeat":repeated})
                print(json.dumps({"generated":i,"total":len(cases),
                                  "status":first["fallback_reason"] or "accepted"}),flush=True)
        write_json(args.output/"summary.json",{
            "questions":len(cases),"fallback_counts":dict(counts),
            "repeats":len(stability),
            "accepted_output_instability":sum(not r["accepted_same"] for r in stability),
            "raw_output_instability":sum(not r["raw_same"] for r in stability),
            "repeat_cases":stability})
        provenance["status"]="complete"
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        model._client.close()
        write_json(args.output/"manifest.json",provenance)

def expansion_queries(original, first):
    expanded=first.get("legal_query")
    if expanded is None:
        return [original]
    if not expanded.startswith(original+" "):
        raise ValueError("expansion must preserve original query")
    return [original,expanded]

def expansion_capture(args):
    from legal_rag.config import load_settings
    from legal_rag.factory import build_hybrid_retriever, close_legs
    settings=load_settings(profile="local",overrides={"paths":{"root":str(args.data_root)}})
    actual=(settings.retrieval.k_dense,settings.retrieval.k_sparse,settings.retrieval.k_fused,
            settings.retrieval.rrf_k,settings.retrieval.dense_weight,settings.retrieval.sparse_weight,
            settings.retrieval.leg_floor,settings.rerank.dtype)
    if actual!=(100,100,100,60,1,.75,3,"float16"):
        raise ValueError("resolved baseline differs from plan: "+repr(actual))
    source=json.loads((args.expansions/"manifest.json").read_text())
    if source.get("status")!="complete" or source.get("target_sha256")!=sha(args.target):
        raise ValueError("expansions are incomplete or target hash differs")
    cases=[json.loads(line) for line in args.target.read_text().splitlines() if line]
    if args.limit: cases=cases[:args.limit]
    expansions={row["case_id"]:row for line in (args.expansions/"expansions.jsonl").read_text().splitlines()
                if line for row in [json.loads(line)]}
    if any(case["case_id"] not in expansions for case in cases):
        raise ValueError("missing frozen expansion")
    depth50_manifest=json.loads((args.depth50/"manifest.json").read_text())
    if depth50_manifest.get("status")!="complete" or depth50_manifest.get("target_sha256")!=sha(args.target):
        raise ValueError("depth-50 control is incomplete or target differs")
    control={row["case_id"]:row["record"]["reranked"]["hits"][:10]
             for line in (args.depth50/"variants.jsonl").read_text().splitlines()
             if line for row in [json.loads(line)] if row["variant"]=="baseline"}
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=manifest(args,settings)
    provenance.update(target_sha256=sha(args.target),
                      expansion_manifest_sha256=sha(args.expansions/"manifest.json"),
                      expansion_file_sha256=sha(args.expansions/"expansions.jsonl"))
    write_json(args.output/"manifest.json",provenance)
    try:
        retriever=build_hybrid_retriever(settings)
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        write_json(args.output/"manifest.json",provenance)
        raise
    rows=[]
    fallback_parity=0
    try:
        for i in range(args.warmup):
            retriever.retrieve(cases[i%len(cases)]["query"],10,None)
        with ((args.output/"stages.jsonl").open("x") as captures,
              (args.output/"per-case.jsonl").open("x") as percase):
            for i,case in enumerate(cases,1):
                first=expansions[case["case_id"]]["first"]
                queries=expansion_queries(case["query"],first)
                started=time.perf_counter()
                rec=capture_multi_stages(retriever,queries,depth=50,rerank_k=100)
                elapsed=(time.perf_counter()-started)*1000
                if errors(rec):
                    append(captures,{"case":case,"expansion":first,"errors":errors(rec)})
                    raise RuntimeError("failed retrieval stage: "+repr(errors(rec)))
                if len(queries)==1:
                    expected=[h["chunk_id"] for h in control[case["case_id"]]]
                    actual=[h["chunk_id"] for h in rec["reranked"]["hits"][:10]]
                    if actual!=expected:
                        raise RuntimeError("fallback does not reproduce original-only depth-50 control")
                    fallback_parity+=1
                union=list({h["chunk_id"]:h for leg in rec["legs"] for h in leg["hits"]}.values())
                result=metrics(case,rec,union)
                result.update(variant="C5",expanded=len(queries)==2,
                              generation_ms=first["elapsed_ms"],retrieval_ms=elapsed,
                              total_ms=first["elapsed_ms"]+elapsed)
                rows.append(result)
                append(percase,result)
                small=compact(rec)
                small["legs"]=[{key:leg[key] for key in
                                ("name","kind","query_index","requested_depth","timing_ms","error")} |
                               {"hits":[{k:v for k,v in hit.items() if k!="chunk"}
                                        for hit in leg["hits"]]} for leg in rec["legs"]]
                append(captures,{"case_id":case["case_id"],"queries":queries,"record":small,
                                 "top10":[h for h in rec["reranked"]["hits"][:10]]})
                print(json.dumps({"captured":i,"total":len(cases),
                                  "expanded":len(queries)==2}),flush=True)
        write_json(args.output/"summary.json",{
            "c5":summarize(rows),"fallback_parity":fallback_parity,
            "expanded_cases":sum(r["expanded"] for r in rows)})
        provenance.update(status="complete",fallback_parity=fallback_parity)
    except BaseException as exc:
        provenance.update(status="failed",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        write_json(args.output/"manifest.json",provenance)
        close_legs(retriever)

def calibration_run(args):
    import csv
    source=json.loads((args.pool_dir/"manifest.json").read_text())
    if source.get("status")!="unjudged":
        raise ValueError("expected an unjudged pool")
    pool=[json.loads(line) for line in (args.pool_dir/"review-pool.jsonl").read_text().splitlines() if line]
    provenance=[json.loads(line) for line in (args.pool_dir/"pool-provenance.jsonl").read_text().splitlines() if line]
    target=[json.loads(line) for line in args.target.read_text().splitlines() if line]
    by_case={row["case_id"]:row["domain"] for row in target}
    if len(by_case)!=source["questions"] or len(pool)!=source["pairs"]:
        raise ValueError("target/pool membership differs from source manifest")
    rows,hidden=sample_review_calibration(pool,provenance,by_case,seed=args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"review-calibration.csv").open("x",newline="",encoding="utf-8-sig") as handle:
        fields=["review_id","case_id","query","chunk_id","document_id","document_type",
                "text","grade","note"]
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output/"calibration-strata.jsonl").open("x") as handle:
        for row in hidden: append(handle,row)
    write_json(args.output/"manifest.json",{
        "status":"awaiting_independent_labels","seed":args.seed,"pairs":len(rows),
        "questions":len({r["case_id"] for r in rows}),"per_domain":25,
        "sampling_proxy_per_domain":{"reference":8,"baseline":9,"variant":8},
        "text_prefix_chars":3500,"pool_sha256":sha(args.pool_dir/"review-pool.jsonl"),
        "pool_provenance_sha256":sha(args.pool_dir/"pool-provenance.jsonl"),
        "target_sha256":sha(args.target),
        "instructions":"An independent reviewer grades each pair 0, 1, or 2 using the plan rubric. Source membership and retrieval origin are hidden in calibration-strata.jsonl and are not relevance labels."})

def compare_c5_rows(groups):
    from legal_rag.evaluation.recall_report import paired_interval
    result=compare_depth_rows(groups)
    def value(row):
        return {field:row[field] for field in ("case_id","group_id","domain")} | {
            "value":row["reference_metrics"]["top10"]["chunk_cutoff_recall"]}
    result["C5"]["vs_depth50_top10"]=paired_interval(
        [value(row) for row in groups["depth50"]],
        [value(row) for row in groups["C5"]])
    return result

def expansion_report(args):
    from statistics import median
    paths={"baseline":args.baseline,"depth50":args.depth50,"C5":args.c5}
    groups={}
    hashes=set()
    index_hashes=set()
    for name,path in paths.items():
        provenance=json.loads((path/"manifest.json").read_text())
        if provenance.get("status")!="complete":
            raise ValueError("incomplete "+name+" run")
        hashes.add(provenance["target_sha256"])
        index_hashes.add(json.dumps(provenance["index_metadata"],sort_keys=True))
        expected="C5" if name=="C5" else "baseline"
        groups[name]=[row for line in (path/"per-case.jsonl").read_text().splitlines()
                      if line for row in [json.loads(line)] if row["variant"]==expected]
    if len(hashes)!=1:
        raise ValueError("different target snapshots")
    if len(index_hashes)!=1:
        raise ValueError("index metadata changed between comparison runs")
    result=compare_c5_rows(groups)
    generation=json.loads((args.expansions/"summary.json").read_text())
    expanded=sum(row["expanded"] for row in groups["C5"])
    if expanded!=generation["fallback_counts"].get("accepted",0):
        raise ValueError("capture/expansion acceptance counts differ")
    latencies=sorted(row["total_ms"] for row in groups["C5"])
    timing={"mean_generation_ms":sum(row["generation_ms"] for row in groups["C5"])/len(groups["C5"]),
            "median_total_evaluator_ms":median(latencies),
            "p95_total_evaluator_ms":latencies[int(.95*(len(latencies)-1))]}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"comparison.json",{
        "decision":"no_adoption_unjudged_development","target_sha256":next(iter(hashes)),
        "runs":result,"generation":generation,"expanded_cases":expanded,
        "fallback_parity":json.loads((args.c5/"summary.json").read_text())["fallback_parity"],
        "timing":timing,"timing_limitation":"Capture serialization is included; this is not serving latency."})
    lines=["# C5 query expansion development result","",
           "Known-source diagnostics on previously used questions; passage relevance remains unjudged.","",
           "| Retrieval | Questions | Macro union source coverage | Macro source R@10 |",
           "| --- | ---: | ---: | ---: |"]
    for name in ("baseline","depth50","C5"):
        m=result[name]["summary"]["macro"]
        lines.append(f"| {name} | {result[name]['questions']} | "
                     f"{m['union.reference_coverage']:.3f} | {m['top10.chunk_cutoff_recall']:.3f} |")
    b=result["C5"]["top10"]
    d=result["C5"]["vs_depth50_top10"]
    lines += ["",f"Paired C5 minus B0 source R@10: {b['delta']:+.3f} "
              f"(exploratory 95% CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]).",
              f"Paired C5 minus depth-50 control: {d['delta']:+.3f} "
              f"(exploratory 95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]).",
              "",f"Accepted expansions: {expanded}/100; fallback cases reproduce the depth-50 original-only control.",
              f"Repeated generation changed the accepted expansion on {generation['accepted_output_instability']}/{generation['repeats']} queries.",
              f"Mean generation {timing['mean_generation_ms']:.1f} ms; evaluator total p50/p95 "
              f"{timing['median_total_evaluator_ms']:.1f}/{timing['p95_total_evaluator_ms']:.1f} ms. "
              "Capture serialization is included, so these are not serving latency estimates.",
              "","No winner is selected without independent passage labels. Novel citations trigger fallback and accepted terms may still be semantically poor."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")

def compare_depth_rows(groups):
    from legal_rag.evaluation.recall_report import paired_interval
    baseline=groups["baseline"]
    if not baseline or any(row["errors"] for rows in groups.values() for row in rows):
        raise ValueError("missing baseline or failed requests")
    result={}
    for name,rows in groups.items():
        if len(rows)!=len(baseline):
            raise ValueError("case membership differs")
        metrics={}
        if name!="baseline":
            for stage,key in (("union","reference_coverage"),("top10","chunk_cutoff_recall")):
                def value(row):
                    return {field:row[field] for field in ("case_id","group_id","domain")} | {
                        "value":row["reference_metrics"][stage][key]}
                metrics[stage]=paired_interval([value(r) for r in baseline],
                                               [value(r) for r in rows])
        result[name]={"questions":len(rows),"summary":summarize(rows),**metrics}
    return result

def depth_report(args):
    paths={"baseline":args.baseline,"depth50":args.depth50,"depth200":args.depth200}
    groups={}
    target_hashes=set()
    for name,path in paths.items():
        provenance=json.loads((path/"manifest.json").read_text())
        if provenance.get("status")!="complete":
            raise ValueError("incomplete "+name+" run")
        target_hashes.add(provenance["target_sha256"])
        groups[name]=[r for line in (path/"per-case.jsonl").read_text().splitlines()
                      if line.strip() for r in [json.loads(line)] if r["variant"]=="baseline"]
    if len(target_hashes)!=1:
        raise ValueError("different target snapshots")
    result={"decision":"no_adoption_unjudged_development",
            "target_sha256":next(iter(target_hashes)),"runs":compare_depth_rows(groups)}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"comparison.json",result)
    lines=["# Candidate depth development comparison","",
           "Previously used questions and known source IDs only. No independently judged relevance or serving change.","",
           "| Depth per dense/sparse leg | Questions | Macro union coverage | Macro source R@10 | Paired R@10 delta vs 100 | 95% CI |",
           "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name,label in (("baseline","100"),("depth50","50"),("depth200","200")):
        row=result["runs"][name]
        macro=row["summary"]["macro"]
        interval=row.get("top10")
        lines.append(f"| {label} | {row['questions']} | {macro['union.reference_coverage']:.3f} | "
                     f"{macro['top10.chunk_cutoff_recall']:.3f} | "+
                     (f"{interval['delta']:+.3f} | [{interval['ci95'][0]:+.3f}, {interval['ci95'][1]:+.3f}] |"
                      if interval else "— | — |"))
    lines += ["","Each depth was a separate live retrieval at the same fused budget of 100; ANN rankings can change with requested depth.",
              "Known-reference IDs are incomplete relevance labels; the result cannot select a serving variant."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")

def report_run(args):
    import csv
    from collections import defaultdict
    from legal_rag.evaluation.recall_report import paired_interval
    provenance=json.loads((args.run_dir/"manifest.json").read_text())
    if provenance.get("status")!="complete":
        raise ValueError("only complete runs can be compared")
    rows=[json.loads(l) for l in (args.run_dir/"per-case.jsonl").read_text().splitlines() if l.strip()]
    if any(r["errors"] for r in rows):
        raise ValueError("failed requests must be resolved before comparison")
    groups=defaultdict(list)
    for row in rows: groups[row["variant"]].append(row)
    baseline={r["case_id"]:r for r in groups["baseline"]}
    if not baseline: raise ValueError("no baseline")
    summaries={name:summarize(items) for name,items in groups.items()}
    comparisons={}
    for name,items in groups.items():
        if name=="baseline": continue
        if len({r["case_id"] for r in items})!=len(items):
            raise ValueError("duplicate variant cases")
        if name!="full_union" and {r["case_id"] for r in items}!=baseline.keys():
            raise ValueError("variant case membership differs")
        def value(row):
            return {k:row[k] for k in ("case_id","group_id","domain")} | {
                "value":row["reference_metrics"]["top10"]["chunk_cutoff_recall"]}
        selected=[baseline[r["case_id"]] for r in items]
        comparisons[name]={"paired_questions":len(items),**paired_interval(
            [value(r) for r in selected],[value(r) for r in items])}
    loss=defaultdict(lambda:{"questions":0,"absent_from_union":0,"lost_at_fusion":0,
                             "lost_at_rerank_top10":0,"lost_during_document_selection":0})
    for row in baseline.values():
        m=row["reference_metrics"]
        counts=loss[row["domain"]]
        counts["questions"]+=1
        u=m["union"]["reference_coverage"]
        f=m["fused"]["reference_coverage"]
        t=m["top10"]["chunk_cutoff_recall"]
        e=m["prepared"]["reference_coverage"]
        counts["absent_from_union"]+=int(u==0)
        counts["lost_at_fusion"]+=int(u>0 and f==0)
        counts["lost_at_rerank_top10"]+=int(f>0 and t==0)
        counts["lost_during_document_selection"]+=int(t>0 and e==0)
    result={"decision":"no_adoption_unjudged_development","summaries":summaries,
            "comparisons":comparisons,"baseline_loss_counts":dict(loss),
            "source_manifest_sha256":sha(args.run_dir/"manifest.json"),
            "source_cases_sha256":sha(args.run_dir/"per-case.jsonl")}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"comparison.json",result)
    with (args.output/"per_case.csv").open("x") as handle:
        fields=["variant","case_id","domain","group_id","union_reference_coverage",
                "fused_reference_coverage","chunk_cutoff_reference_recall10","prepared_reference_coverage"]
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            m=row["reference_metrics"]
            writer.writerow({k:row[k] for k in fields[:4]} | {
                "union_reference_coverage":m["union"]["reference_coverage"],
                "fused_reference_coverage":m["fused"]["reference_coverage"],
                "chunk_cutoff_reference_recall10":m["top10"]["chunk_cutoff_recall"],
                "prepared_reference_coverage":m["prepared"]["reference_coverage"]})
    text=["# Development recall results","",
          "These are known-reference diagnostics on previously used questions. Relevance judgments and fresh confirmation are pending; no serving change is adopted.","",
          "Recall@10 below counts labeled reference documents found in the first 10 chunks. A domain's values are averaged over its questions; the headline gives each domain equal weight.","",
          "| Variant | Questions | Macro reference R@10 | Paired delta vs B0 | 95% CI |",
          "| --- | ---: | ---: | ---: | --- |"]
    for name,summary in summaries.items():
        score=summary["macro"].get("top10.chunk_cutoff_recall",0)
        c=comparisons.get(name)
        text.append(f"| {name} | {summary['requests']} | {score:.3f} | "+
                    (f"{c['delta']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |" if c else "— | — |"))
    text += ["","Full-union results use only their fixed 40-question subset; their delta compares against B0 on those same questions. Do not compare that row's absolute score directly with the 100-question rows.",
             "","## Baseline stage losses","",
             "| Domain | Questions | Reference absent from union | Lost at fusion | Lost at top-10 reranking | Lost at document selection |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for domain,c in loss.items():
        text.append("| "+domain+" | "+" | ".join(str(c[k]) for k in c)+" |")
    text += ["","Loss counts describe complete loss of known reference IDs, not loss of every relevant authority. Document selection does not establish passage survival: that needs validated supporting spans.",
             "","Fusion comparisons reuse fixed candidates and pointwise reranker scores. Their replay timings are not end-to-end serving latency. Confidence intervals are exploratory and are not adjusted for selecting among many variants.",
             "","Next gate: independently judge the shared top-10 pool before choosing a finalist. A fresh confirmation set remains unavailable."]
    (args.output/"report.md").write_text("\n".join(text)+"\n")
    print(json.dumps({"output":str(args.output),"variants":len(groups),"decision":result["decision"]}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="operation",required=True)
    for name in ("audit","capture","ann"):
        command=sub.add_parser(name)
        command.add_argument("--data-root",type=Path,required=True)
        command.add_argument("--output",type=Path,required=True)
        if name=="ann":
            command.add_argument("--target",type=Path,required=True)
            command.add_argument("--efs",nargs="*",type=int,default=[])
        if name=="capture":
            command.add_argument("--target",type=Path,required=True)
            command.add_argument("--limit",type=int,default=0)
            command.add_argument("--depth",type=int,default=100)
            command.add_argument("--parity",type=int,default=20)
            command.add_argument("--warmup",type=int,default=10)
            command.add_argument("--variants",nargs="+",choices=[*VARIANTS,"R0"],default=["baseline"])
            command.add_argument("--full-union",action="store_true")
    agreement_cmd=sub.add_parser("calibration-agreement")
    agreement_cmd.add_argument("--calibration",type=Path,required=True)
    agreement_cmd.add_argument("--judge-run",type=Path,required=True)
    agreement_cmd.add_argument("--reviewed-csv",type=Path,required=True)
    agreement_cmd.add_argument("--reviewer",required=True)
    agreement_cmd.add_argument("--reviewer-type",choices=["human","ai"],required=True)
    agreement_cmd.add_argument("--output",type=Path,required=True)
    judge=sub.add_parser("judge-calibration")
    judge.add_argument("--calibration",type=Path,required=True)
    judge.add_argument("--output",type=Path,required=True)
    judge.add_argument("--base-url",default="http://localhost:11434")
    judge.add_argument("--limit",type=int,default=0)
    r3=sub.add_parser("r3-report")
    r3.add_argument("--baseline",type=Path,required=True)
    r3.add_argument("--output",type=Path,required=True)
    c5report=sub.add_parser("expansion-report")
    for name in ("baseline","depth50","c5","expansions","output"):
        c5report.add_argument("--"+name,type=Path,required=True)
    c5=sub.add_parser("expansion-capture")
    c5.add_argument("--data-root",type=Path,required=True)
    c5.add_argument("--target",type=Path,required=True)
    c5.add_argument("--expansions",type=Path,required=True)
    c5.add_argument("--depth50",type=Path,required=True)
    c5.add_argument("--output",type=Path,required=True)
    c5.add_argument("--limit",type=int,default=0)
    c5.add_argument("--warmup",type=int,default=10)
    generate=sub.add_parser("expansion-generate")
    generate.add_argument("--data-root",type=Path,required=True)
    generate.add_argument("--target",type=Path,required=True)
    generate.add_argument("--output",type=Path,required=True)
    generate.add_argument("--limit",type=int,default=0)
    calibration=sub.add_parser("calibration")
    calibration.add_argument("--pool-dir",type=Path,required=True)
    calibration.add_argument("--target",type=Path,required=True)
    calibration.add_argument("--output",type=Path,required=True)
    calibration.add_argument("--seed",type=int,default=20260928)
    extension_r3=sub.add_parser("pool-extend-r3")
    for name in ("pool-dir","baseline","r3","output"):
        extension_r3.add_argument("--"+name,type=Path,required=True)
    extension=sub.add_parser("pool-extend")
    for name in ("pool-dir","baseline","c5","output"):
        extension.add_argument("--"+name,type=Path,required=True)
    pool=sub.add_parser("pool")
    pool.add_argument("--run-dir",type=Path,required=True)
    pool.add_argument("--data-root",type=Path,required=True)
    pool.add_argument("--output",type=Path,required=True)
    depth=sub.add_parser("depth-report")
    for name in ("baseline","depth50","depth200","output"):
        depth.add_argument("--"+name,type=Path,required=True)
    reporter=sub.add_parser("report")
    reporter.add_argument("--run-dir",type=Path,required=True)
    reporter.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.operation=="judge-calibration" and args.limit<0:
        parser.error("negative limit")
    if args.operation=="expansion-capture" and (args.limit<0 or args.warmup<0):
        parser.error("negative limit or warmup")
    if args.operation=="expansion-generate" and args.limit<0:
        parser.error("negative limit")
    if args.operation=="ann" and any(ef<1 for ef in args.efs):
        parser.error("--efs values must be positive")
    if args.operation=="capture" and (args.depth<1 or args.limit<0 or args.parity<0 or args.warmup<0):
        parser.error("invalid negative limit or nonpositive depth")
    if args.operation=="capture" and args.depth!=100 and args.parity:
        parser.error("baseline parity requires depth 100; set --parity 0 for depth variants")
    {"audit":audit,"capture":capture,"ann":ann,"report":report_run,
     "depth-report":depth_report,"pool":pool_run,
     "calibration":calibration_run,
     "expansion-generate":expansion_generate,
     "expansion-capture":expansion_capture,
     "expansion-report":expansion_report,
     "pool-extend":pool_extend,"pool-extend-r3":pool_extend_r3,
     "r3-report":r3_report,"judge-calibration":judge_calibration,
     "calibration-agreement":calibration_agreement}[args.operation](args)

if __name__=="__main__":
    main()
