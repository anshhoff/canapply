from canapply.eligibility import Profile, judge
from canapply.requirements import Requirements

STUDENT_IN = Profile(country="IN", work_auth=["IN"], years_experience=1, is_student=True, grad_year=2027)


def R(**kw) -> Requirements:
    return Requirements(**kw)


def bucket(**kw) -> str:
    return judge(R(**kw), STUDENT_IN).bucket


def test_job_in_own_country_with_no_stated_bar_is_can_apply():
    assert bucket(work_mode="onsite", job_countries=["IN"], level="new_grad") == "can_apply"


def test_one_year_short_is_stretch_and_names_the_gap():
    v = judge(R(work_mode="onsite", job_countries=["IN"], min_years=2, min_years_quote="2+ years"), STUDENT_IN)
    assert v.bucket == "stretch"
    assert v.blockers[0].name == "experience" and v.blockers[0].quote == "2+ years"


def test_big_experience_gap_is_cannot():
    assert bucket(work_mode="onsite", job_countries=["IN"], min_years=4) == "cannot"
    assert bucket(work_mode="onsite", job_countries=["IN"], min_years=3) == "cannot"      # exactly 2 short


def test_within_half_a_year_counts_as_meeting_it():
    almost = STUDENT_IN.model_copy(update={"years_experience": 0.8})
    assert judge(R(work_mode="onsite", job_countries=["IN"], min_years=1), almost).bucket == "can_apply"
    assert judge(R(work_mode="onsite", job_countries=["IN"], min_years=2), almost).bucket == "stretch"


def test_senior_title_counts_even_without_a_number():
    assert bucket(work_mode="onsite", job_countries=["IN"], level="senior") == "cannot"


def test_remote_limited_to_another_country_is_cannot():
    v = judge(R(work_mode="remote", remote_scope="countries", remote_countries=["US"], job_countries=["US"]), STUDENT_IN)
    assert v.bucket == "cannot" and v.blockers[0].name == "location"


def test_remote_worldwide_is_can_apply():
    assert bucket(work_mode="remote", remote_scope="anywhere", level="junior") == "can_apply"


def test_remote_region_must_cover_the_country():
    assert bucket(work_mode="remote", remote_scope="region", remote_regions=["APAC"], level="junior") == "can_apply"
    assert bucket(work_mode="remote", remote_scope="region", remote_regions=["EMEA"], level="junior") == "cannot"


def test_remote_with_no_scope_falls_back_to_listed_country():
    assert bucket(work_mode="remote", remote_scope="unknown", job_countries=["US"]) == "cannot"
    assert bucket(work_mode="remote", remote_scope="unknown", job_countries=["IN"], level="junior") == "can_apply"


def test_abroad_onsite_visa_not_mentioned_is_stretch():
    v = judge(R(work_mode="onsite", job_countries=["GB"], level="new_grad"), STUDENT_IN)
    assert v.bucket == "stretch" and v.blockers[0].name == "work_auth"


def test_abroad_with_no_sponsorship_is_cannot():
    assert bucket(work_mode="onsite", job_countries=["GB"], work_auth="must_already_have") == "cannot"
    assert bucket(work_mode="onsite", job_countries=["US"], work_auth="citizens_only") == "cannot"


def test_abroad_with_sponsorship_is_can_apply():
    assert bucket(work_mode="onsite", job_countries=["SG"], work_auth="sponsorship_offered", level="new_grad") == "can_apply"


def test_two_soft_misses_is_cannot():
    assert bucket(work_mode="onsite", job_countries=["GB"], min_years=2) == "cannot"


def test_us_internship_without_work_rights_is_cannot():
    assert bucket(work_mode="onsite", job_countries=["US"], employment="internship", level="intern") == "cannot"


def test_enrollment_in_another_country_is_cannot():
    assert bucket(work_mode="onsite", job_countries=["IN"], employment="internship", enrollment_country="SG") == "cannot"


def test_phd_required_is_cannot_and_masters_is_stretch():
    assert bucket(work_mode="onsite", job_countries=["IN"], degree="phd") == "cannot"
    assert bucket(work_mode="onsite", job_countries=["IN"], degree="masters", level="new_grad") == "stretch"


def test_profile_changes_the_answer():
    r = R(work_mode="onsite", job_countries=["GB"], work_auth="must_already_have", level="new_grad")
    assert judge(r, STUDENT_IN).bucket == "cannot"
    assert judge(r, STUDENT_IN.model_copy(update={"work_auth": ["IN", "GB"]})).bucket == "can_apply"


def test_unknowns_do_not_block_but_are_reported():
    v = judge(R(work_mode="onsite", job_countries=["IN"]), STUDENT_IN)
    assert v.bucket == "can_apply" and any(c.name == "experience" for c in v.unverified)
