import re

re_age = re.compile(r"[_ ]{1}[p|P]{0,1}[\d]{1,3}[d]{0,1}[_ ]{1}", re.IGNORECASE)
re_splfile = re.compile(r"^(?P<datetime>([\d]{8}-[\d]{4}))-(?P<type>([(n)(p)(SPL)])).txt$")
re_click_file = re.compile(r"^(?P<datetime>([\d]{8}-[\d]{4}))-n.txt$")
re_pyabr3_click_file = re.compile(
    r"^(?P<datetime>([\d]{4}-[\d]{2}-[\d]{2}))_click_(?P<serial>([0-9]{3}_[0-9]{3})).p$"
)
re_tone_file = re.compile(r"^(?P<datetime>([\d]{8}-[\d]{4}))-n-(?P<frequency>([0-9.]*)).txt$")
re_pyabr3_interleaved_tone_file = re.compile(
    r"^(?P<datetime>([\d]{4}-[\d]{2}-[\d]{2}))_interleaved_plateau_(?P<serial>([0-9]{3}_[0-9]{3})).p$"
)
re_subject = re.compile(
    r"(?P<strain>([A-Za-z0-9]{3,16}))[_ ]{1}(?P<sex>([M|F]{1}))[_ ]{1}(?P<subject>([A-Z0-9]{1,5}))[_ ]{1}(?P<age>([p|P\d]{1,4}))[_ ]{1}(?P<date>([0-9]{0,8}))[_ ]{0,1}(?P<treatment>([A-Za-z]{2}))"
)


