import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import textwrap
import sys
import io
import matplotlib
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Dict, Any, Optional
from typing import Tuple
from generate_answers.datascience.prompts import data_analyst_prompt, explainability_prompt
from generate_answers.datascience.data_info import get_dataframe_info
from generate_answers.datascience.llm_response import answer_question
from generate_answers.llm_clients import LLMPipeline

def load_data(filepath: str) -> Tuple[pd.DataFrame, str]:
    data = pd.read_csv(filepath)
    data_sample = pd.read_csv(filepath).head(5)
    data_string = data_sample.to_string(index=False)
    return (data, data_string)

def format_explanation_prompt(user_question: str, dataset_info: Dict[str, Any], code: str, answer: str, model_name: str, temperature: float, pipeline: LLMPipeline) -> str:
    final_template = explainability_prompt.format(user_question=user_question, dataset_info=dataset_info, code=code, answer=answer)
    return answer_question(final_template, pipeline, model_name, temperature)

def analyze_csv_stream(model_name: str, temperature: float, filepath: str, query: str, pipeline: LLMPipeline) -> Dict[str, Any]:
    csv_file_path = filepath
    user_query = query

    # Save the original matplotlib backend
    original_backend = matplotlib.get_backend()
    
    try:
        # Set matplotlib to use non-interactive backend
        matplotlib.use('Agg')
        
        data, csv_data_string = load_data(filepath)
        namespace = {'data': data, 'pd': pd, 'plt': plt, 'sns': sns, 'np': np}
        csv_data_info = get_dataframe_info(data)

        formatted_plan_prompt = data_analyst_prompt.format(
            user_query=query,
            csv_data_info=csv_data_info,
            csv_data_string=csv_data_string
        )

        plan_response = answer_question(formatted_plan_prompt, pipeline, model_name, temperature)

        try:
            clean_response = plan_response.replace('```json', '').replace('```', '').strip()
            analysis_plan = json.loads(clean_response)
        except Exception as e:
            print(f"JSON parsing failed: {str(e)}")
            # print(f"Raw response: {plan_response}")
            return {"error": f"JSON parsing failed: {str(e)}"}

        if 'step' not in analysis_plan:
            return {"error": "JSON response missing 'step' key"}

        if not isinstance(analysis_plan['step'], list):
            return {"error": "'step' should be a list/array"}

        if len(analysis_plan['step']) == 0:
            return {"error": "'step' array is empty"}

        accumulated_code = ""
        Response_Analysis = ""
        all_outputs = []
        all_plots = []
        Reasoning = []

        for i, step in enumerate(analysis_plan['step'], 1):
            if 'code' not in step:
                all_outputs.append(f"Error: Step {i} missing 'code' field")
                continue

            step_code = step['code']
            cleaned_step_code = textwrap.dedent(step_code)
            accumulated_code += f"\nCODE: {cleaned_step_code}\n"
            Reasoning.append(step.get('explanation', 'No explanation provided'))

            try:
                # Capture stdout and stderr while executing the code
                captured_output = StringIO()
                captured_stderr = StringIO()
                
                with redirect_stdout(captured_output), redirect_stderr(captured_stderr):
                    exec(cleaned_step_code, globals(), namespace)
                
                # Get the captured output
                step_output = captured_output.getvalue().strip()
                step_stderr = captured_stderr.getvalue().strip()
                
                # Combine stdout and stderr if both exist
                if step_output and step_stderr:
                    combined_output = f"{step_output}\nSTDERR: {step_stderr}"
                elif step_stderr:
                    combined_output = f"STDERR: {step_stderr}"
                else:
                    combined_output = step_output

                if not combined_output:
                    combined_output = str(namespace.get('answer', ''))

                all_outputs.append(f"Step {i} Output:\n{combined_output}")
                
                # Close all plots to free memory
                plt.close('all')
                
            except Exception as e:
                all_outputs.append(f"Step {i} Error: {str(e)}")

        Response_Code = accumulated_code
        Response_Output = "\n---------------------\n".join(all_outputs)
        Response_Reasoning = '\n'.join([f"### STEP {i+1} ### : {text}" for i, text in enumerate(Reasoning)])

        if "Error:" not in Response_Output:
            Response_Analysis = format_explanation_prompt(
                query, csv_data_info, Response_Code, Response_Output, model_name, temperature, pipeline
            )

        if not Response_Analysis:
            Response_Analysis = "Could not generate analysis due to code execution failure."

        return {
            'Response_Code': Response_Code,
            'Response_Output': Response_Output,
            'Response_Analysis': Response_Analysis,
            'Response_Reasoning': Response_Reasoning
        }

    except FileNotFoundError:
        return {"error": f"File not found at path: {csv_file_path}"}
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}
    finally:
        # Restore the original matplotlib backend
        matplotlib.use(original_backend)
        plt.close('all')