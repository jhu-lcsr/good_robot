KEY_PREFIX = ""
CURRENT_STATES = {}


def get_current_meters():
    return CURRENT_STATES


def log_value(key, value):
    #prefix = get_run_name() + "/" + KEY_PREFIX + "/"
    #key = prefix + key
    CURRENT_STATES[key] = value


def reset():
    global CURRENT_STATES
    CURRENT_STATES = {}


def set_prefix(prefix):
    global KEY_PREFIX
    KEY_PREFIX = prefix