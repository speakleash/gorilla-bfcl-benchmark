from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override
import json
import re

from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing

class BielikHandlerFC(OSSHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, dtype="bfloat16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype, **kwargs)

    @override
    def _format_prompt(self, messages, function):
        """
        "bos_token": "<s>",
        "chat_template": "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        """

        formatted_prompt = "<s>"

        system_content = self.get_funcall_sys_prompt()
        system_content += "\n\n" + "\n\n".join([json.dumps(tool, ensure_ascii=False) for tool in function])
        formatted_prompt += f"<|im_start|>system\n{system_content}<|im_end|>\n"

        for message in messages:
            formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += f"<|im_start|>assistant\n"
        
        return formatted_prompt

    def _extract_tool_calls_from_text(self, text: str) -> list[dict]:
        """
        Extract tool calls from text, handling both:
        - With <tool_call>...</tool_call> tags
        - Without tags (raw JSON)
        
        Returns a list of parsed tool call dictionaries.
        """
        tool_calls = []
        
        # First, try to extract tool calls wrapped in <tool_call> tags
        tag_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tag_matches = re.findall(tag_pattern, text, re.DOTALL)
        
        if tag_matches:
            # Found tool calls with tags
            for match in tag_matches:
                try:
                    parsed = json.loads(match.strip())
                    tool_calls.append(parsed)
                except json.JSONDecodeError:
                    continue
        else:
            # No tags found, try parsing as raw JSON
            # First, try parsing the entire text as a single JSON object
            stripped_text = text.strip()
            try:
                parsed = json.loads(stripped_text)
                tool_calls.append(parsed)
            except json.JSONDecodeError:
                # Try splitting by newlines and parsing each line
                for line in stripped_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Try to find JSON objects in the line
                    # Handle cases where JSON might be surrounded by whitespace
                    try:
                        parsed = json.loads(line)
                        tool_calls.append(parsed)
                    except json.JSONDecodeError:
                        # Try to extract JSON from the line using regex
                        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}'
                        json_matches = re.findall(json_pattern, line)
                        for json_match in json_matches:
                            try:
                                parsed = json.loads(json_match)
                                tool_calls.append(parsed)
                            except json.JSONDecodeError:
                                continue
        
        return tool_calls

    def _convert_to_ast_format(self, tool_calls: list[dict]) -> list[dict]:
        """Convert parsed tool calls to AST format: [{name: arguments}, ...]"""
        return [{tc["name"]: tc["arguments"]} for tc in tool_calls if "name" in tc and "arguments" in tc]

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if isinstance(result, str):
            tool_calls = self._extract_tool_calls_from_text(result)
            if tool_calls:
                result = self._convert_to_ast_format(tool_calls)
            else:
                # Return empty list if no valid tool calls found
                result = []
        elif isinstance(result, list):
            processed_results = []
            for sublist in result:
                inner_result = []
                for item in sublist:
                    if isinstance(item, str):
                        tool_calls = self._extract_tool_calls_from_text(item)
                        inner_result.extend(self._convert_to_ast_format(tool_calls))
                processed_results.append(inner_result)
            result = processed_results
        else:
            return result

        return result

    @override
    def decode_execute(self, result, has_tool_call_tag):
        function_call_list = self._extract_tool_calls_from_text(result)
        
        execution_list = []
        for function_call in function_call_list:
            key = function_call['name']
            value = function_call['arguments']
            if isinstance(value, str):
                value = json.loads(value)
            execution_list.append(
                f"{key}({','.join([f'{k}={repr(v)}' for k,v in value.items()])})"
            )
        return execution_list
    
    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        self.test_category: str = test_entry["id"].rsplit("_", 1)[0]

        # override the default bfcl system prompt, Bielik uses its own system prompt
        
        # To be verified if needed
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        
        return {"message": [], "function": functions}
    
    @staticmethod
    def get_funcall_sys_prompt():
        return """
You are provided with tool signatures that you can use to assist with the user's query. You do not have to use a tool if you can respond adequately without it. Do not make assumptions about the values to use in tool calls. If the user's message is missing required parameters or if you do not have an appropriate tool to fulfill the request, inform the user or ask for clarification instead of attempting to call any tools.

If you decide to invoke a tool, you MUST use the following JSON format:  
`<tool_call>{"name": <tool-name>, "arguments": <args-dict>}</tool_call>`

Below is a list of tools in JSON format that you can invoke:
    """.strip()

