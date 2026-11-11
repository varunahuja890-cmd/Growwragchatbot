"""PII guard — block before any log, embed, or artifact write (§8)."""

from groww_faq.query.pii import PII_MESSAGE, detect_pii


def test_pan_blocks():
    assert detect_pii("My PAN is ABCDE1234F, check my folio") == "pan"


def test_aadhaar_blocks():
    assert detect_pii("My Aadhaar is 1234 5678 9012") == "aadhaar"
    assert detect_pii("Aadhaar 123456789012") == "aadhaar"


def test_email_blocks():
    assert detect_pii("reach me at varun@example.com please") == "email"


def test_mobile_blocks():
    assert detect_pii("call me on 9876543210") == "phone"


def test_otp_blocks():
    assert detect_pii("OTP 482193") == "otp"


def test_account_number_blocks():
    # 11 digits: long enough for the account pattern, short enough that
    # it is not also an aadhaar-shaped 12-digit run.
    assert detect_pii("account number 12345678901") == "account"


def test_aadhaar_shaped_account_number_still_blocks():
    # A 12-digit "account number" is aadhaar-shaped; the kind label may
    # differ but the turn must be blocked either way.
    assert detect_pii("account number 123456789012") is not None


def test_normal_fact_question_passes():
    for question in [
        "What is the expense ratio of HDFC Large Cap?",
        "What is the exit load of the HDFC ELSS Tax Saver Fund?",
        "What is the lock-in period of the ELSS?",
        "What is the benchmark of HDFC Mid Cap?",
    ]:
        assert detect_pii(question) is None, question


def test_message_is_the_fixed_line():
    assert PII_MESSAGE == "I don't need or store personal details."
