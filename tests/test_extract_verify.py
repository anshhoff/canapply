"""The code checks that sit between the LLM and the database. No network, no model."""
from canapply.extract import by_rules, merge, verify
from canapply.requirements import Requirements
from canapply.search import rrf

JOB = {
    "title": "Backend Engineer", "company": "Acme", "location_raw": "Remote - United States", "workplace_raw": "Remote",
    "employment_raw": "Full-time",
    "description": "About the role\nYou will build APIs.\nRequirements\n- 3+ years of experience with Python\n"
                   "- You must be authorized to work in the United States; we are unable to sponsor visas.",
}


def test_true_quotes_survive():
    r = Requirements(min_years=3, min_years_quote="3+ years of experience with Python", work_auth="must_already_have",
                     work_auth_quote="we are unable to sponsor visas", work_mode="remote", remote_scope="countries",
                     remote_countries=["US"], location_quote="Remote - United States")
    clean, total, ok = verify(r, JOB)
    assert (total, ok) == (3, 3) and clean.min_years == 3 and clean.work_auth == "must_already_have"


def test_invented_quote_is_dropped():
    r = Requirements(min_years=7, min_years_quote="7+ years of experience leading teams")
    clean, total, ok = verify(r, JOB)
    assert clean.min_years is None and (total, ok) == (1, 0)


def test_real_quote_with_wrong_number_is_dropped():
    r = Requirements(min_years=5, min_years_quote="3+ years of experience with Python")   # quote exists, number does not
    assert verify(r, JOB)[0].min_years is None


def test_remote_anywhere_needs_a_worldwide_word():
    r = Requirements(work_mode="remote", remote_scope="anywhere", location_quote="Remote - United States")
    clean, _, ok = verify(r, JOB)
    assert ok == 0 and clean.remote_scope == "countries" and clean.remote_countries == ["US"]
    world = dict(JOB, location_raw="Remote, Worldwide")
    r2 = Requirements(work_mode="remote", remote_scope="anywhere", location_quote="Remote, Worldwide")
    assert verify(r2, world)[0].remote_scope == "anywhere"


def test_rules_reading_of_the_same_posting():
    r = by_rules(JOB)
    assert r.min_years == 3 and r.work_mode == "remote" and r.remote_countries == ["US"]
    assert r.work_auth == "must_already_have"


def test_merge_fills_gaps_from_board_facts():
    m = merge(by_rules(JOB), Requirements())
    assert m.job_countries == ["US"] and m.work_mode == "remote" and m.employment == "full_time"


def test_rrf_rewards_agreement_between_arms():
    scores = rrf({1: 1, 2: 2, 3: 3}, {3: 1, 4: 2, 1: 3})
    ranked = sorted(scores, key=lambda i: -scores[i])
    assert ranked[:2] == [1, 3] or ranked[:2] == [3, 1]       # in both lists beats top of one list
    assert scores[2] > 0 and 4 in scores
