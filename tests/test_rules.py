from canapply import geo, rules
from canapply.textutil import html_to_text, quote_in_text


def test_title_level():
    assert rules.level_from_title("Software Engineer Intern (Summer 2027)") == "intern"
    assert rules.level_from_title("New Grad Software Engineer") == "new_grad"
    assert rules.level_from_title("Senior Backend Engineer") == "senior"
    assert rules.level_from_title("Staff Machine Learning Engineer") == "senior"
    assert rules.level_from_title("Engineering Manager, Payments") == "senior"
    assert rules.level_from_title("Junior Data Analyst") == "junior"
    assert rules.level_from_title("Backend Engineer") == "unknown"


def test_is_tech():
    assert rules.is_tech("Machine Learning Engineer")
    assert rules.is_tech("Forward Deployed Engineer")
    assert not rules.is_tech("Account Executive, Enterprise")
    assert not rules.is_tech("Senior Counsel, Privacy")
    assert rules.is_tech("Sales Engineer")            # 'engineer' wins over 'sales'


def test_years_required_takes_lowest_required():
    text = "Requirements\n- 3+ years of experience building backend services\n- 5+ years of experience preferred"
    years, quote = rules.years_required(text)
    assert years == 3 and "3+ years" in quote


def test_years_required_ignores_non_requirements():
    assert rules.years_required("We were founded 10 years ago and have grown fast.")[0] is None
    assert rules.years_required("Equity vests over 4 years with a 1 year cliff.")[0] is None
    assert rules.years_required("At least two years of professional experience in Python.")[0] == 2


def test_work_auth_signals():
    s = rules.work_auth_signals("You must be authorized to work in the United States. We are unable to sponsor visas.")
    assert "no_sponsorship" in s and "auth_required" in s
    assert "citizenship" in rules.work_auth_signals("Applicants must be a U.S. citizen due to export control rules.")
    assert rules.work_auth_signals("We love remote work.") == {}


def test_parse_location():
    assert geo.parse_location("Bengaluru, India")[0] == ["IN"]
    assert geo.parse_location("Remote - United States") == (["US"], [], True)
    assert geo.parse_location("Austin, TX")[0] == ["US"]
    assert set(geo.parse_location("London | Berlin | New York")[0]) == {"GB", "DE", "US"}
    countries, regions, remote = geo.parse_location("Remote, EMEA")
    assert countries == [] and regions == ["EMEA"] and remote
    assert geo.parse_location(None) == ([], [], False)


def test_html_to_text_handles_escaped_html():
    out = html_to_text("&lt;p&gt;We need &lt;strong&gt;3+ years&lt;/strong&gt;.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;")
    assert "We need 3+ years." in out and "- Python" in out and "<" not in out


def test_quote_must_be_verbatim():
    text = "Requirements:\n- 3+ years of experience   with Python\n- Must be authorised to work in the UK"
    assert quote_in_text("3+ years of experience with Python", text)        # whitespace differences are fine
    assert quote_in_text("must be authorised to work in the uk", text)      # case differences are fine
    assert not quote_in_text("5+ years of experience with Python", text)    # a changed number is not
    assert not quote_in_text("Requires a US work visa", text)               # invented text is not
    assert not quote_in_text("", text) and not quote_in_text(None, text)


def test_ingest_plan_writes_only_new_and_changed_postings():
    from canapply.ingest import plan
    rows = [{"ext_id": "a", "content_hash": "1"}, {"ext_id": "b", "content_hash": "2"}, {"ext_id": "c", "content_hash": "3"}]
    write, same, added = plan(rows, {"a": "1", "b": "old"})
    assert [r["ext_id"] for r in write] == ["b", "c"]      # changed text, and brand new
    assert same == ["a"] and added == 1
