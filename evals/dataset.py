"""The golden set: pure data, no imports, so importing it can never run a model.

Every expected_answer is a short checkable fact -- a number, a name, a date --
rather than a sentence, because correctness here is scored by containment in
code, not by a model's opinion. `accept` lists the other spellings of the same
fact that must count as right (4,500 / 4500), so scoring never punishes
formatting.

`expected_source` names the fixture a correct answer must come from, which is
what makes retrieval separately checkable from generation: an answer can be
right for the wrong reason, and this catches that.

Unanswerable cases carry expected_source=None and a `forbidden` list. Declining
is the correct behaviour; inventing a specific figure is the failure mode worth
measuring, since a RAG agent that never says "not found" is a hallucination
machine.
"""

POLICY = "nordvale_scholarship_policy.txt"
FAQ = "nordvale_admissions_faq.txt"
HELIX = "helix_labs_internship_brief.txt"

GOLDEN_SET = [
    {
        "id": "award-amount",
        "question": "How many credits per year does the Nordvale Merit Scholarship award?",
        "expected_answer": "4,500",
        "accept": ["4,500", "4500"],
        "expected_source": POLICY,
        "unanswerable": False,
    },
    {
        "id": "deadline",
        "question": "On what date do Nordvale Merit Scholarship applications close?",
        "expected_answer": "14 March",
        "accept": ["14 march", "march 14"],
        "expected_source": POLICY,
        "unanswerable": False,
    },
    {
        "id": "statement-limit",
        "question": "What is the word limit for the Nordvale Merit Scholarship personal statement?",
        "expected_answer": "800 words",
        "accept": ["800"],
        "expected_source": POLICY,
        "unanswerable": False,
    },
    {
        "id": "reference-letters",
        "question": "How many reference letters must a Nordvale Merit Scholarship applicant submit?",
        "expected_answer": "two",
        "accept": ["two", "2"],
        "expected_source": POLICY,
        "unanswerable": False,
    },
    {
        "id": "committee-chair",
        "question": "Who chairs the Nordvale scholarship committee?",
        "expected_answer": "Dr. Imelda Farrow",
        "accept": ["farrow"],
        "expected_source": POLICY,
        "unanswerable": False,
    },
    {
        "id": "minimum-gpa",
        "question": "What is the minimum GPA Nordvale requires for consideration?",
        "expected_answer": "3.2",
        "accept": ["3.2"],
        "expected_source": FAQ,
        "unanswerable": False,
    },
    {
        "id": "interview-month",
        "question": "In which month does Nordvale hold admissions interviews, and are they online or in person?",
        "expected_answer": "April, online",
        "accept": ["april"],
        "expected_source": FAQ,
        "unanswerable": False,
    },
    {
        "id": "admissions-contact",
        "question": "What email address should I use to contact the Nordvale admissions office?",
        "expected_answer": "admissions@nordvale.example",
        "accept": ["admissions@nordvale.example"],
        "expected_source": FAQ,
        "unanswerable": False,
    },
    {
        "id": "decision-timing",
        "question": "How many days after the interview does Nordvale release decisions?",
        "expected_answer": "30 days",
        "accept": ["30"],
        "expected_source": FAQ,
        "unanswerable": False,
    },
    {
        "id": "deferral",
        "question": "How long can a Nordvale place be deferred for?",
        "expected_answer": "one year",
        "accept": ["one year", "1 year"],
        "expected_source": FAQ,
        "unanswerable": False,
    },
    {
        "id": "residency-duration",
        "question": "How many weeks does the Helix Labs Summer Residency run for?",
        "expected_answer": "12 weeks",
        "accept": ["12"],
        "expected_source": HELIX,
        "unanswerable": False,
    },
    {
        "id": "residency-stipend",
        "question": "What monthly stipend do Helix Labs residents receive?",
        "expected_answer": "2,100 credits",
        "accept": ["2,100", "2100"],
        "expected_source": HELIX,
        "unanswerable": False,
    },
    {
        "id": "mentor-week",
        "question": "In which week of the Helix Labs residency is a mentor assigned?",
        "expected_answer": "week 2",
        "accept": ["week 2", "second week"],
        "expected_source": HELIX,
        "unanswerable": False,
    },
    {
        # Two documents: the limit is only stated in the Helix brief as an offset
        # from a number that appears solely in the Nordvale policy.
        "id": "cross-document-limit",
        "question": "In words, what is the Helix Labs application statement limit?",
        "expected_answer": "600 words",
        "accept": ["600"],
        "expected_source": HELIX,
        "also_requires_source": POLICY,
        "unanswerable": False,
    },
    {
        "id": "unanswerable-fee",
        "question": "What is the application fee for the Nordvale Merit Scholarship?",
        "expected_answer": "Not stated in the documents.",
        "accept": [],
        "expected_source": None,
        "unanswerable": True,
        # Narrow on purpose: a correct decline may legitimately mention the
        # 4,500 credit award while saying no fee is stated.
        "forbidden": ["fee is ", "fee of "],
    },
    {
        "id": "unanswerable-legal-head",
        "question": "Who is the head of the Helix Labs legal department?",
        "expected_answer": "Not stated in the documents.",
        "accept": [],
        "expected_source": None,
        "unanswerable": True,
        "forbidden": ["the head of the legal", "legal department is"],
    },
]
