#### link evaluation metric for each method on the test dataset presentnin json format
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
)
from deepeval.test_case import LLMTestCaseParams
import os
from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset
import pytest
from deepeval import assert_test
import pandas as pd 
load_dotenv('keys.env')
os.environ["OPENAI_API_KEY"]  = os.getenv('OPENAI_API_KEY')

def get_mean(df, metric_list = [{'Contextual Precision': 'Contextual Precision Score'}, {'Contextual Recall': 'Contextual Recall Score'}, {'Contextual Relevancy': 'Contextual Relevancy Score'}, {'Answer Relevancy': 'Answer Relevancy Score'}, {'Faithfulness': 'Faithfulness Score'}, {'Hallucination': 'Hallucination Score'}]):
    aggregate_scores = {}
    # print(metric_list)
    for metric_dict in metric_list:
        for k, val in metric_dict.items():
            aggregate_scores[k] = df[val].mean()
    return aggregate_scores

def convert_to_csv(test_results, file_name: str):
    data = []
    metric_list = []
    iter = 0
    for test in test_results:
        row = {
                'Test Success': test.success,
                'Test Input': test.input,
                'Test Actual Output': test.actual_output,
                'Test Expected Output': test.expected_output,
                'Test Context':  test.context,
                'Test Retrieval Context': test.retrieval_context,
        }
        for metric in test.metrics_data:
            if iter==0:
                metric_list.append({f'{metric.name}':f'{metric.name} Score'})
            dict = {
                f'{metric.name} Threshold': metric.threshold,
                f'{metric.name} Success': metric.success,
                f'{metric.name} Score': metric.score,
                f'{metric.name} Reason': metric.reason,
            }
            row.update(dict)
        data.append(row)
        iter = iter+1
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")
    aggregate_scores = get_mean(df,metric_list)
    return aggregate_scores


def evaluate_methods_(method_name):
    outfile = f"{method_name}.csv"
    outfile = "./eval_methods/"+outfile
    try:
        structured_response = {
            "method":" ".join(method_name.split("_")[:2]),
            "evaluation":"",
            "evaluation_file":""
        }
    except:
        structured_response = {
            "method":method_name,
            "evaluation":"",
            "evaluation_file":""
        }
    if os.path.exists(outfile):
        print("just reading the file and obtaining the aggregate scores.")
        df = pd.read_csv(outfile)
        structured_response["evaluation"] = get_mean(df)
        structured_response["evaluation_file"] = outfile
    else:
        dataset = EvaluationDataset()
        dataset.add_test_cases_from_json_file(
            file_path=f"./eval_methods/{method_name}.json",
            input_key_name="input",
            expected_output_key_name="expected_output",
            actual_output_key_name="actual_output",
            context_key_name="context",
            retrieval_context_key_name="retrieval_context",
        )
        test_cases = dataset._llm_test_cases
        contextual_precision = ContextualPrecisionMetric()
        contextual_recall = ContextualRecallMetric()
        contextual_relevancy = ContextualRelevancyMetric()
        hallucination = HallucinationMetric(threshold=0.5)
        answer_relevancy = AnswerRelevancyMetric()
        faithfulness = FaithfulnessMetric()

        eval_out = evaluate(
            test_cases=test_cases,
            metrics=[
                contextual_precision,
                contextual_recall,
                contextual_relevancy,
                answer_relevancy,
                faithfulness,
                hallucination
            ]
        )
        aggregate_scores = convert_to_csv(eval_out, outfile)
        structured_response["evaluation"] = aggregate_scores
        structured_response["evaluation_file"] = outfile
    return structured_response

# out = evaluate_methods("method_4_without_heading_index")
# out = evaluate_methods("method_5_without_doc_filter")