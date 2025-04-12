import pyttsx3

engine = pyttsx3.init()

def speak(text):
    print("\n🔊 Speaking...")
    engine.say(text)
    engine.runAndWait()
