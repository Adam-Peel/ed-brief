"""Generate RSS fixtures from real recent headlines, dated relative to now.

Regenerate with:  python tests/make_fixtures.py
Dates are relative so the recency scoring stays exercised however old the
checkout is.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape

FIXTURES = Path(__file__).parent / "fixtures"

FEEDS = {
    "bbc-education": (
        "BBC Education",
        [
            ("'Relentless' GCSE resit cycle 'simply not working', school and college leaders say",
             "School and college leaders say the requirement to resit GCSE maths and English is demoralising students and consuming teaching time.", 6),
            ("Students get their GCSE and BTec grades as app launches for 750 schools in England",
             "Hundreds of thousands of teenagers receive results as the government's new digital credentials app goes live.", 8),
            ("Competition for degree apprenticeships quadruples as university costs rise",
             "Applications per place have risen sharply as students weigh tuition fees against earning while training.", 20),
            ("Teacher recruitment falls short of secondary targets for a fourth year",
             "Department for Education figures show initial teacher training recruitment reaching only two thirds of the secondary target, with history among the shortage subjects.", 30),
        ],
    ),
    "schools-week": (
        "Schools Week",
        [
            ("Curriculum review: what the Francis report means for history teaching",
             "The Curriculum and Assessment Review recommends changes to the national curriculum programmes of study, with implications for how history is sequenced at key stage 3.", 12),
            ("Cost-cutters' advice could be released in written form",
             "Schools facing budget pressure may receive written recommendations from the department's school resource management advisers.", 18),
            ("Ofsted report cards: inspectors to trial new grading in autumn",
             "The inspectorate confirms pilot schools for its replacement for single-word judgements under the revised inspection framework.", 26),
            ("Early career framework reforms: what changes for ECTs in September",
             "Induction arrangements and mentor training are revised, with implications for early career teachers starting this year.", 40),
            ("Every pupil must feel they are valued and that they can contribute",
             "A headteacher writes on building belonging and inclusion across a diverse intake.", 50),
        ],
    ),
    "fe-week": (
        "FE Week",
        [
            ("GCSE resit results: 15.3% of post-16 maths students achieve grade 4 or above",
             "The English pass rate fell to 19.8% despite increased entries among post-16 learners.", 7),
            ("College principals warn on adult education budget allocations",
             "Funding settlements leave providers unable to plan beyond the current year.", 44),
        ],
    ),
    "dfe": (
        "DfE (gov.uk)",
        [
            ("£1bn boost to PE and school sport to end fitness postcode lottery",
             "New funding for school sport facilities and coaching across primary and secondary schools.", 5),
            ("Government sets out next steps for children's social care reforms",
             "Reforms cover kinship care, family help and information sharing between agencies.", 22),
            ("New bursaries announced for history and modern languages trainee teachers",
             "Initial teacher training bursaries are extended to cover further shortage subjects from the next recruitment cycle.", 34),
        ],
    ),
    "ofsted": (
        "Ofsted (gov.uk)",
        [
            ("Ofsted publishes revised school inspection handbook",
             "The handbook sets out how the new inspection framework will operate, including changes to deep dives in foundation subjects.", 16),
        ],
    ),
    "guardian-education": (
        "Guardian Education",
        [
            ("Teaching about empire: schools grapple with a contested history curriculum",
             "History departments describe navigating debate over how colonial history and slavery are taught at key stage 3.", 14),
            ("Vice-chancellor pay rises again as university funding crisis deepens",
             "Analysis of higher education accounts shows senior pay outpacing staff settlements.", 28),
            ("Cognitive science in the classroom: what the evidence actually supports",
             "Researchers caution that retrieval practice and cognitive load theory are often oversimplified in school CPD.", 36),
        ],
    ),
}

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>{name}</title>
<link>https://example.invalid/{fid}</link>
<description>Fixture feed for testing</description>
{items}
</channel></rss>
"""

ITEM = """<item>
<title>{title}</title>
<link>https://example.invalid/{fid}/{n}</link>
<description>{summary}</description>
<pubDate>{date}</pubDate>
<guid>https://example.invalid/{fid}/{n}</guid>
</item>"""


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    for fid, (name, entries) in FEEDS.items():
        items = "\n".join(
            ITEM.format(
                title=escape(title),
                summary=escape(summary),
                fid=fid,
                n=n,
                date=format_datetime(now - timedelta(hours=age)),
            )
            for n, (title, summary, age) in enumerate(entries)
        )
        path = FIXTURES / f"{fid}.xml"
        path.write_text(
            TEMPLATE.format(name=escape(name), fid=fid, items=items), encoding="utf-8"
        )
        print(f"wrote {path.name} ({len(entries)} items)")


if __name__ == "__main__":
    main()
