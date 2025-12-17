textvqa_demonstrations = {
    "version": 0.0,
    "demonstrations": {
        {"what is the brand of phone?":
             ["Thought: first find the phone. \nAction: Region Ground Module \nAction Input: Where is the phone? \nObservation: Region found",
              "Thought: read the logo and text on the phone. \nAction: OCR Module \nAction Input: What texts are printed on the phone? \nObservation: Nokia",
              "Thought: find which information indicate its brand. \nAction: Language Module \nAction Input: What is the brand of phone? \nObservation: Nokia"],
         "what type of plane is this?":
             ["Thought: first find the plane. \nAction: Region Ground Module \nAction Input: Where is the plane? \nObservation: Region found",
              "Thought: read the logo and text on the plane. \nAction: OCR Module \nAction Input: What texts are printed on the plane? \nObservation: ALE",
              "Thought: infer the type of plane for the given information \nAction: Language Module \nAction Input: what type of plane is this? \nObservation: ALE"],
         "what are the letters on the tail section of the plane?":
             ["Thought: focus on the plane tail. \nAction: Region Ground Module \nAction Input: Where is the tail section of the plane? \nObservation: Region found",
              "Thought: read letters on it. \nAction: OCR Module \nAction Input: What letters are printed on the tail section of the plane? \nObservation: F"],
         "who is this copyrighted by?":
             ['Action: OCR Module \nAction Input: What texts are shown in the picture? \nObservation: Alex Messi, 2020',
              'Action: Language Module \nAction Input: who is this copyrighted by? \n Observation: Alex Messi'],
         "what year is shown in the photo?":
             ["Action: OCR Module \nAction Input: What texts are shown in the photo?",
              'Action: Language Module \nAction Input: What year is shown in the photo?'],
         "what time does the sales meeting start?":
             ["Action: OCR Module \nAction Input: What texts are shown?",
              "Action: Language Module \nAction Input: what time does the sales meeting start?"],
         "what app is shown on the right phone?":
             ["Action: Region Ground Module \nAction Input: Where is the right phone?",
              "Action: OCR Module \nAction Input: What texts are shwon on the right phone?",
              "Action: Language Module \nAction Input: What app is shown on the right phone?"],
         "how long has the drink on the right been aged?":
             ["Action: Region Ground Module \nAction Input: Where is the drink on the right?",
              "Action: OCR Module \nAction Input: What texts are shwon on it?",
              "Action: Language Module \nAction Input: What app is shown on the right phone?"],
         "what color is the number 19 on the player's jersey?": [
             "Action: Region Ground Module \nAction Input: Where is the player?",
             "Action: OCR Ground Module \nAction Input: Where is the text '19'?",
             "Action: Caption Module \nAction Input: What color is the number 19?"],
         "what color are the letters on this sign?":
             ["Action: Region Ground Module \nAction Input: Where is the sign?",
              "Action: OCR Ground Module \nAction Input: Where is the letters?",
              "Action: Caption Module \nAction Input: What color are the letters?"],
         "what color is the beer?":
             ["Action: Region Ground Module \nAction Input: Where is the beer?",
              "Action: Caption Module \nAction Input: What color is the beer?"],
         "what type of liquor is displayed?":
             ["Action: Region Ground Module \nAction Input: Where is the liquor?",
              "Action: OCR Module \nAction Input: What texts are printed on the liquor?",
              "Action: Language Module \nAction Input: what type of liquor is displayed?"],
         "What number is on the player's jersey?": [
             "Action: Region Ground Module \nAction Input: Where is the player's jersey?",
             "Action: OCR Module \nAction Input: What texts are printed on it?",
             "Action: Language Module \nAction Input: What number is on the player's jersey?"],
         "What name is next to number 5?":
             ["Action: OCR Module \nAction Input: What is printed on the photo?",
              "Action: Language Module \nAction Input: What name is next to number 5?"],
         }
    }
}
