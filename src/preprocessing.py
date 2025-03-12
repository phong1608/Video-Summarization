from wtpsplit import SaT



class Preprocessor:
    def __init__(self):
        self.Sat = SaT
    
    def tokenize(self,paragraph):
        sat = SaT("sat-3l")
       
        sat.half().to("cuda")

        sentences=sat.split(paragraph)
        return sentences