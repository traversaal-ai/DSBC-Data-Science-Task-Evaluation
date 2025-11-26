prompt_start = """
Respond in this exact JSON format:
{
  {
    "Evaluation": 'Yes' 'No' 'The Evaluation should be Yes only when the response is absolutely correct. For numerical values, the rounded .2f values should match to be considered correct.',
  }
}
"""
prompt_end = """
    You are being used for LLM-as-judge. In numeric solutions an error beyond the 2nd decimal point after rounding can be ignored.
    The parsing might have an issue, in that case you can also look at the analysis/reasoning to judge whether the response is written somewhere in there.
    ####
    The Query by the user is :
    {Q}
    ----
    The Ground Truth for the query is :
    {A}
    ----
    The Code Snippet to obtain the ground truth was :
    {C}
    ----
    The Response by the model is :
    {R}
    ----
    The Code Snippet in the submission was :
    {S}
    ----
    The Reasoning given with the submission was :
    {E}
    """