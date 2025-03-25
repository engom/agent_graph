import asyncio
import json
import math
import re
import subprocess
from asyncio import Semaphore
from functools import lru_cache

import boto3
import numexpr
from botocore.config import Config

# from langchain_aws import ChatBedrock
from langchain_core.tools import BaseTool, StructuredTool, tool

# Limit concurrent requests
sem = Semaphore(5)

# Create a global boto3 client to reuse connections
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    config=Config(retries=dict(max_attempts=3), connect_timeout=5, read_timeout=30),
)


async def generate_(bedrock_runtime, model_id, system_prompt, messages):
    async with sem:  # Using the existing semaphore
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages,
                "temperature": 0.5,
            }
        )

        response = await asyncio.to_thread(
            bedrock_runtime.invoke_model, body=body, modelId=model_id
        )
        response_body = json.loads(response.get("body").read())
        return response_body


@lru_cache(maxsize=100)
def generate_code(query: str) -> str:
    """Generates Python code for a given natural language task using SolveBio Expression syntax rules"""
    query = query.strip().replace("\n", "")

    prompt = """You are an expert in generating Python code for EDP/SolvBio.
    Follow these SolveBio Expression syntax rules:
    1. Basic Structure:
    - Expressions are string-based formulas
    - Support for context variables without declaration

    2. Data Types:
    - string: Use double quotes for string literals
    - integer: Direct numbers (1, 2, 3)
    - float: Decimal numbers (1.0, 2.5)
    - boolean: true, false
    - list: [1, 2, 3] or ["a", "b", "c"]

    3. Operators:
    - Arithmetic: +, -, *, /, %
    - Comparison: ==, !=, >, <, >=, <=
    - Logical: and, or, not
    - String concat: + (for strings)

    4. Built-in Functions:
    - len(): Length of strings or lists
    - min(), max(): For numbers or strings
    - sum(): For numeric lists
    - round(): For numbers
    - range(): For creating number sequences

    5. Field References:
    - Direct reference: field_name
    - No $ or special prefixes needed

    The code should:
    - Follow SolveBio syntax rules (NO 'def' or 'return')
    - Include proper type casting when necessary
    - Handle null values gracefully

    Output only the code, no explanations.
    """
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    system_prompt = prompt
    user_message = {"role": "user", "content": query}
    messages = [user_message]

    try:
        # Run the async function in a new event loop
        response = asyncio.run(
            generate_(bedrock_runtime, model_id, system_prompt, messages)
        )

        code_gen = response.get("content")[0].get("text")
        print("Code generator output code:")
        print(code_gen)

        return code_gen.strip()
    except Exception as e:
        return f"Error generating code: {str(e)}"


# @tool
def calculator(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator)
calculator.name = "calculator"


description = """Generate EDP/SolvBio (Entreprise Data Platform) code from natural language queries.

    Useful for when you need to address questions about data processing tasks using SolveBio.
    This function converts natural language queries into SolveBio-compatible expressions,
    which are Python-like formulas used for data manipulation and analysis.

    Args:
        query (str): A natural language description of the desired data processing task.
                    For example: "Convert the start dates from the ex Dataset to the proper format for further derivations."

    Returns:
        str: A valid EDP expression that can be executed. The expression follows SolveBio
             syntax rules and can include:
             - Basic Python operations and built-in functions
             - Dataset queries and field access
             - Statistical calculations
             - Type casting and data transformations

    Supported Features:
        - Single-line expressions (multi-line for readability only)
        - Built-in Python functions (len, min, max, sum, round, range)
        - SolveBio-specific functions (dataset_field_stats, etc.)
        - Data type handling (string, text, date, integer, float, boolean, object)
        - List operations and transformations

    Raises:
        subprocess.TimeoutExpired: If expression execution exceeds 10 seconds
        subprocess.CalledProcessError: If the generated code fails to execute

    Examples:
        >>> query = "Convert the start dates from the ex Dataset to the proper format for further derivations."
        >>> result = generate_code(query)
        >>> print(result)
        'datetime_format(record.ex_startdate, output_format="%d%b%Y") if record.ex_startdate else None'

        >>> query = "Translate VISITC values to VISITN using visit code list Dataset."
        >>> result = generate_code(query)
        >>> print(result)
        'get(dataset_field_values(
            "path/to/visit_timepoint_codelist",
            field="visitn",
            filters=[["visitc", record.visitc]]
        ), 0)'

        >>> query = "Create a column Race based on the values of different race-related variables."
        >>> result = generate_code(query)
        >>> print(result)
        (
            '"AMERICAN INDIAN/ALASKA NATIVE" if record.raceame == 1 else '
            '"ASIAN" if record.raceasi == 1 else '
            '"BLACK/AFRICAN AMERICAN" if record.racebla == 1 else '
            '"NATIVE HAWAIIAN/PACIFIC ISLANDER" if record.racenat == 1 else '
            '"WHITE" if record.racewhi == 1 else '
            '"NOT REPORTED" if record.racentre == 1 else '
            'None'
        )
    """

# Update the StructuredTool configuration
code_generator = StructuredTool.from_function(
    func=generate_code,
    name="code_generator",
    description=description,
)


# # @tool
# def test_code(query: str) -> str:
#     """Tests a given code and output results"""
#     try:
#         # Read the generated code
#         with open("./temp.py", "r") as file:
#             code_to_test = file.read()

#         # Generate test code with SolveBio-specific test cases
#         test_prompt = f"""
#         Create a test script for SolveBio Expression that:
#         1. Imports necessary modules:
#            from solvebio import Expression

#         2. Creates sample test cases with these patterns:
#            # Test static expressions
#            expr = Expression('your_expression')
#            result = expr.evaluate(data_type='appropriate_type', is_list=False)

#            # Test with context variables
#            expr = Expression('expression_with_fields')
#            data = {{'field1': 'value1', 'field2': 'value2'}}
#            result = expr.evaluate(data=data, data_type='appropriate_type', is_list=False)

#         3. Tests the following code with appropriate inputs:

#         {code_to_test}

#         Output only the test code, no explanations.
#         """

#         test_result = llm_.invoke(test_prompt)

#         # Extract and clean test code
#         test_code = test_result.content
#         if "```python" in test_code:
#             test_code = test_code.split("```python")[1].split("```")[0]
#         elif "```" in test_code:
#             test_code = test_code.split("```")[1]

#         test_code = test_code.strip()

#         # Save test code
#         with open("./temp-test.py", "w") as file_test:
#             file_test.write(test_code)

#         # Execute test with timeout and capture output
#         result = subprocess.run(
#             ["python", "temp-test.py"],
#             capture_output=True,
#             text=True,
#             timeout=10,
#             check=True,
#         )

#         return result.stdout

#     except subprocess.TimeoutExpired:
#         return "Error: Code execution timed out after 10 seconds"

#     except subprocess.CalledProcessError as e:
#         return f"Error executing code: {e.stderr}"

#     except Exception as e:
#         return f"Error testing code: {str(e)}"

#     finally:
#         # Clean up temp files
#         subprocess.run(["rm", "temp.py", "temp-test.py"], capture_output=True)


# if __name__ == "__main__":
#     code_gen = generate_code(
#         "Convert the start dates from the ex Dataset to the proper format for further derivations."
#     )
#     print(code_gen)
#     # print(test_code())
