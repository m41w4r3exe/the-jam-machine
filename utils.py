from miditok import Event


def writeToFile(path, content):
    if type(content) is not str:
        content = str(content)
    with open(path, "w") as f:
        f.write(content)


# Function to read from text from txt file:
def readFromFile(path):
    with open(path, "r") as f:
        return f.read()


class TextToEvent:
    def getlist(self, type, value):
        event_type = str(type).lower()
        try:
            return getattr(self, event_type, "")(value)
        except Exception as e:
            print("Error: Unknown event", type, value)
            raise e

    def time_shift(self, value):
        return [Event("Time-Shift", value)]

    def note_on(self, value):
        return [Event("Note-On", value), Event("Velocity", 100)]

    def note_off(self, value):
        return [Event("Note-Off", value)]

    def velocity(self, value):
        return []

    def chord(self, value):
        return []
