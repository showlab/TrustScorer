imageqa_demonstrations = {
    "version": 0.0,
    "demonstrations":
        {
            "what is the brand of phone?":
            [
                "Question: what is the brand of phone?",
                "visual-0: an image, user-provided image, a phone on the ground.",
                'Thought: first find the phone. \nAction: region_ground("Where is the phone?", [0]) \nObservation: Region found',
                "visual-1: an image, cropped image, the phone.",
                'Thought: read the logo and text on the phone. \nAction: text_recognition(None, visual[0]) \nObservation: Nokia',
                'Thought: find which information indicate its brand. \nAction: language_module("What is the brand of phone?", visual[]) \nObservation: Nokia',
                "Final Answer: Nokia"],

            "what type of plane is this?":
            [
                "Question: what type of plane is this?",
                 "visual-0: an image, user-provided image, a plane on the air.",
                 'Thought: first find the plane. \nAction: region_ground("Where is the plane?", [0]) \nObservation: Region found',
                 "visual-1: an image, cropped image, the plane.",
                 'Thought: read the logo and text on the plane. \nAction: text_recognition(None, [0]) \nObservation: ALE',
                 'Thought: infer the type of plane for the given information \nAction: language_module("what type of plane is this?", visual[]) \nObservation: ALE'],

            "what type of liquor on the left is displayed?":
                ["Question: what type of liquor on the left is displayed?",
                 "visual-0: two bottles on the shelf.",
                 'Thought: first find the liquor on the left. \nAction: region_ground("Where is the liquor on the left?", [0]) \nObservation: Region found',
                 "visual-1: an image, cropped image, the liquor on the left.",
                 "Thought: read the logo and text on the liquor. \nAction: text_recognition(None, visual[0]) \nObservation: Jack Daniel's, 2002, 40% alc/vol",
                 'Thought: infer the type of liquor for the given information \nAction: language_module("what type of liquor on the left is displayed?", visual[]) \nObservation: Jack Daniel'
                 "Final Answer: Jack Daniel's"],

            "What best describes the pool of water? You must choose one answer from: frozen, fresh, dirty, boiling.":
                ['''Question: What best describes the pool of water? You must choose one answer from: frozen, fresh, dirty, boiling. visual-0: an image, user provided image, main content is giraffes standing under a tree
                Thought: Check the visual information about the pool of water in the image.
                Action: image_caption("What best describes the pool of water in the image?", [0])
                Observation: a pond
                Thought: I need more information about the pond to determine its condition. Check the choice one by one.
                Action: image_caption("Is the pool frozen? Is the pool fresh? Is the pool dirty? Is the pool boiling?", [0])
                Observation: The answer for each question is no, no, yes, no.
                Thought: I know the final answer. 
                Final Answer: dirty
                ''']
        },
    "type2question": {'What is': ['the brand of phone?'],
                      'What type': ['of plane is this?', 'of liquor on the left is displayed?'],
                      },
}
