CODE_REPO_TEMPLATE = {}

CODE_REPO_TEMPLATE["REPO_DESCRIPTION"] = """This is a code repository that contains code about a python module named *{module_name}*.
It has several python source code files and documentation files. Here is the list of the files: {files}.
"""

CODE_REPO_TEMPLATE["CODE_DESCRIPTION"] = """This is a python source code file of the python module named *{module_name}*.
The name of the file is {filename}. The path of the file is {filepath}.
"""

CODE_REPO_TEMPLATE["CODE_IMPORTS"] = """This is a code chunk from the python source code file *{filename}* of the python module named *{module_name}*.
It contains import information of the file. The content of the chunk is as follows:
```
{code_chunk_content}
```
"""

CODE_REPO_TEMPLATE["CODE_CHUNK"] = """This is a code chunk from the python source code file *{filename}* of the python module named *{module_name}*.
The content of the chunk is as follows:
```
{code_chunk_content}
```
"""

CODE_REPO_TEMPLATE["DOC_DESCRIPTION"] = """This is a documentation file of the python module named *{module_name}*.
The name of the file is {filename}. The path of the file is {filepath}.
"""

CODE_REPO_TEMPLATE["DOC_CHUNK"] = """This is a text chunk from the documentation file *{filename}* of the python module named *{module_name}*.
The content of the chunk is as follows:
```
{text_chunk_content}
```
"""

CODE_REPO_CONFIGS = {}