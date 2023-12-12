from Rasa_Process import *
from GUI_Connection import *


if __name__ == "__main__":
    while(1):
        button = Wake_up()
        if button == 1:
            button = 0
            while(1):
                record_audio(audio_file, duration=5, sample_rate=16000)
                mess = speech2text(audio_file)
                SendtoGUI(mess)
                
                if mess == '':
                    continue

                response = get_rasa_response(mess)
                for i in range(len(response)):
                    txt = list(response[i].values())[1]
                    SendtoGUI(txt)
                    text2speech(txt)