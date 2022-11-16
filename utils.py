def writeToFile(path, content):
    if type(content) is not str:
        content = str(content)
    with open(path, "w") as f:
        f.write(content)


class EventToText:
    def string(self, event):
        event_type = str(event.type).lower().replace("-", "_")
        return getattr(self, event_type, "")(event.value)

    def time_shift(self, value):
        return f"TIME_SHIFT={value} "

    def note_on(self, value):
        return f"NOTE_ON={value} "

    def note_off(self, value):
        return f"NOTE_OFF={value} "

    def velocity(self, value):
        return ""

    def chord(self, value):
        return ""
