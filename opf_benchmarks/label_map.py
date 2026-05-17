"""Benchmark-label -> OPF-category mapping tables.

Each map keys benchmark-specific labels to one of OPF's 8 categories, or to
``None`` (out of OPF's training scope). Out-of-scope labels are kept in the
``_full`` JSONL with their original label so ``ground_truth_label_recall``
breakdowns remain meaningful, and dropped from the ``_opfscope`` JSONL so
typed evaluation is well-defined.

The AI4Privacy table is a 1:1 port of OpenAI's published PII-Masking-300k
mapping (Privacy Filter Model Card, April 2026, §7.2.1) where labels overlap;
notably ``ip -> private_url``, ``username -> private_person``, and
``otp -> secret``. The Argilla and Nemotron tables apply the same principles
(GPS/coordinate spans -> private_address; IP/IPv4/IPv6 -> private_url;
username/handle -> private_person) so the three benchmarks share one
methodology. Hardware-asset identifiers (MAC, vehicle/device IDs, license
plates, biometric IDs) remain out of scope because OpenAI's mapping does not
include them.
"""

OPF_CATEGORIES = {
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "private_date",
    "private_url",
    "account_number",
    "secret",
}


ARGILLA_TO_OPF: dict[str, str | None] = {
    "FIRSTNAME": "private_person",
    "LASTNAME": "private_person",
    "MIDDLENAME": "private_person",
    "PREFIX": "private_person",
    "EMAIL": "private_email",
    "PHONE": "private_phone",
    "PHONE_NUMBER": "private_phone",
    "PHONENUMBER": "private_phone",
    "PHONEIMEI": "private_phone",
    "STREET": "private_address",
    "STREETADDRESS": "private_address",
    "BUILDINGNUMBER": "private_address",
    "CITY": "private_address",
    "COUNTY": "private_address",
    "STATE": "private_address",
    "ZIPCODE": "private_address",
    "SECONDARYADDRESS": "private_address",
    "DATE": "private_date",
    "DOB": "private_date",
    "TIME": "private_date",
    "ACCOUNTNUMBER": "account_number",
    "CREDITCARDNUMBER": "account_number",
    "CREDITCARDCVV": "account_number",
    "CREDITCARDISSUER": "account_number",
    "MASKEDNUMBER": "account_number",
    "BIC": "account_number",
    "IBAN": "account_number",
    "BITCOINADDRESS": "account_number",
    "LITECOINADDRESS": "account_number",
    "ETHEREUMADDRESS": "account_number",
    "SSN": "account_number",
    "PIN": "secret",
    "PASSWORD": "secret",
    "URL": "private_url",
    "USERNAME": "private_person",
    "USERAGENT": None,
    "IP": "private_url",
    "IPV4": "private_url",
    "IPV6": "private_url",
    "MAC": None,
    "AGE": None,
    "GENDER": None,
    "HEIGHT": None,
    "JOBTITLE": None,
    "JOBAREA": None,
    "JOBTYPE": None,
    "VEHICLEVIN": None,
    "VEHICLEVRM": None,
    "NEARBYGPSCOORDINATE": "private_address",
    "ORDINALDIRECTION": None,
    "CURRENCY": None,
    "CURRENCYSYMBOL": None,
    "CURRENCYCODE": None,
    "CURRENCYNAME": None,
    "AMOUNT": None,
    "SEX": None,
    "SEXTYPE": None,
    "EYECOLOR": None,
    "COMPANYNAME": None,
    "ACCOUNTNAME": None,
}


AI4PRIVACY_TO_OPF: dict[str, str | None] = {
    "FIRSTNAME": "private_person",
    "LASTNAME": "private_person",
    "LASTNAME1": "private_person",
    "LASTNAME2": "private_person",
    "LASTNAME3": "private_person",
    "MIDDLENAME": "private_person",
    "PREFIX": "private_person",
    "GIVENNAME": "private_person",
    "GIVENNAME1": "private_person",
    "GIVENNAME2": "private_person",
    "SURNAME": "private_person",
    "TITLE": "private_person",
    "EMAIL": "private_email",
    "TEL": "private_phone",
    "PHONE": "private_phone",
    "PHONENUMBER": "private_phone",
    "BOD": "private_date",
    "DATE": "private_date",
    "TIME": "private_date",
    "DATEOFBIRTH": "private_date",
    "STREET": "private_address",
    "BUILDING": "private_address",
    "BUILDINGNUMBER": "private_address",
    "SECADDRESS": "private_address",
    "CITY": "private_address",
    "STATE": "private_address",
    "COUNTRY": "private_address",
    "POSTCODE": "private_address",
    "ZIPCODE": "private_address",
    "GEOCOORD": "private_address",
    "IDCARD": "account_number",
    "PASSPORT": "account_number",
    "DRIVERLICENSE": "account_number",
    "SOCIALNUMBER": "account_number",
    "TAXNUM": "account_number",
    "ACCOUNTNUMBER": "account_number",
    "CREDITCARD": "account_number",
    "CREDITCARDNUMBER": "account_number",
    "CREDITCARDCVV": "account_number",
    "IBAN": "account_number",
    "BIC": "account_number",
    "BITCOINADDRESS": "account_number",
    "PASSWORD": "secret",
    "PASS": "secret",
    "PIN": "secret",
    "URL": "private_url",
    "USERNAME": "private_person",
    "USERAGENT": None,
    "IP": "private_url",
    "IPV4": "private_url",
    "IPV6": "private_url",
    "MAC": None,
    "SEX": None,
    "GENDER": None,
    "AGE": None,
    "HEIGHT": None,
    "EYECOLOR": None,
    "JOBTITLE": None,
    "JOBTYPE": None,
    "JOBAREA": None,
    "COMPANYNAME": None,
    "VEHICLEVIN": None,
    "VEHICLEVRM": None,
    "RELIGION": None,
    "NATIONALITY": None,
    "ETHNICITY": None,
    "POLITICAL": None,
    "OTP": "secret",
    "CARDEXPIRY": "private_date",
    "BANKMUNICIP": "private_address",
    "BANKPOSTCODE": "private_address",
    "BANKSTREET": "private_address",
    "BANKNUM": "account_number",
    "DOCNUM": "account_number",
    "CRYPTOADDRESS": "account_number",
}


NEMOTRON_TO_OPF: dict[str, str | None] = {
    "first_name": "private_person",
    "last_name": "private_person",
    "middle_name": "private_person",
    "full_name": "private_person",
    "email": "private_email",
    "phone_number": "private_phone",
    "fax_number": "private_phone",
    "street_address": "private_address",
    "city": "private_address",
    "county": "private_address",
    "state": "private_address",
    "postcode": "private_address",
    "country": "private_address",
    "address": "private_address",
    "date_of_birth": "private_date",
    "date": "private_date",
    "url": "private_url",
    "ssn": "account_number",
    "account_number": "account_number",
    "credit_card": "account_number",
    "cvv": "account_number",
    "bank_routing_number": "account_number",
    "swift_bic": "account_number",
    "iban": "account_number",
    "customer_id": "account_number",
    "employee_id": "account_number",
    "medical_record_number": "account_number",
    "health_plan_beneficiary_number": "account_number",
    "certificate_license_number": "account_number",
    "tax_id": "account_number",
    "password": "secret",
    "pin": "secret",
    "api_key": "secret",
    "user_name": "private_person",
    "ipv4": "private_url",
    "ipv6": "private_url",
    "mac_address": None,
    "coordinate": "private_address",
    "device_identifier": None,
    "vehicle_identifier": None,
    "license_plate": None,
    "biometric_identifier": None,
    "blood_type": None,
    "age": None,
    "gender": None,
    "race_ethnicity": None,
    "sexuality": None,
    "religious_belief": None,
    "political_view": None,
    "occupation": None,
    "education_level": None,
    "credit_debit_card": "account_number",
    "date_time": "private_date",
    "time": "private_date",
    "http_cookie": None,
    "company_name": None,
    "employment_status": None,
    "language": None,
    "unique_id": "account_number",
}


GRETEL_TO_OPF: dict[str, str | None] = {
    # Personal names
    "first_name": "private_person",
    "last_name": "private_person",
    "middle_name": "private_person",
    "name": "private_person",
    "full_name": "private_person",
    "user_name": "private_person",
    # Contact
    "email": "private_email",
    "phone_number": "private_phone",
    "fax_number": "private_phone",
    # Address
    "street_address": "private_address",
    "address": "private_address",
    "city": "private_address",
    "state": "private_address",
    "country": "private_address",
    "postcode": "private_address",
    "coordinate": "private_address",
    # Dates
    "date": "private_date",
    "date_of_birth": "private_date",
    "date_time": "private_date",
    "time": "private_date",
    # Account / financial / government IDs
    "ssn": "account_number",
    "national_id": "account_number",
    "tax_id": "account_number",
    "account_number": "account_number",
    "credit_card_number": "account_number",
    "cvv": "account_number",
    "swift_bic": "account_number",
    "bank_routing_number": "account_number",
    "customer_id": "account_number",
    "employee_id": "account_number",
    "medical_record_number": "account_number",
    "health_plan_beneficiary_number": "account_number",
    "certificate_license_number": "account_number",
    "unique_identifier": "account_number",
    # Secrets
    "password": "secret",
    "pin": "secret",
    "api_key": "secret",
    # URL / network
    "url": "private_url",
    "ipv4": "private_url",
    "ipv6": "private_url",
    # Out of OPF scope
    "mac_address": None,
    "device_identifier": None,
    "vehicle_identifier": None,
    "license_plate": None,
    "biometric_identifier": None,
    "company_name": None,
}


MAPS = {
    "argilla": ARGILLA_TO_OPF,
    "ai4privacy": AI4PRIVACY_TO_OPF,
    "nemotron": NEMOTRON_TO_OPF,
    "gretel": GRETEL_TO_OPF,
}


def map_label(benchmark: str, label: str) -> str | None:
    """Return the OPF category for a benchmark label, or None if out of scope.

    Unknown labels (not in the table) return None and are treated as
    out-of-scope; the adapter logs them so the mapping table can be extended.
    """
    table = MAPS[benchmark]
    return table.get(label)


def is_known_label(benchmark: str, label: str) -> bool:
    return label in MAPS[benchmark]
