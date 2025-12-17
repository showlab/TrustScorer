nextqa_demonstrations = {
    "version": 0.0,
    "demonstrations":
        {"why did the boy pick up one present from the group of them and move to the sofa?":
            [
'''Question: why did the boy pick up one present from the group of them and move to the sofa? You must choose one answer from: share with the girl, approach lady sitting there, unwrap it, playing with toy train, gesture something
The visual inputs are as follows:
visual-0: a 90.02 seconds video, None, user provided video, a boy standing in front of a couch playing a video game 

Thought: I need to know more information about what does the boy in the video do. 
Action: Video Caption Module
Action Input: (What did the boy do?, [0])
Observation: ...

Thought: I need to know infer why did the boy pick up one present from the group of them and move to the sofa?
Action: LLM Module
Acton Input: (why did the boy pick up one present from the group of them and move to the sofa?, [])
Observation: ...

Thought: I know the final answer.
Final Answer: unwrap it
'''],
        "what is the brand of phone?":
            []
        },
    "type2question": {'What is': ['the brand of phone?'],
                      'What type': ['of plane is this?', 'of liquor on the left is displayed?'],
                      },
}
