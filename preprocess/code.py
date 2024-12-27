from tree_sitter import Language, Parser
import tree_sitter_python as tspy
import tree_sitter_markdown as tsmd
import tree_sitter as ts
from typing import List, Tuple, Union
from .template import CODE_REPO_TEMPLATE
from pprint import pprint
import os
import json

PY_LANGUAGE = Language(tspy.language())
MD_LANGUAGE = Language(tsmd.language())

testfilename = "./test.py"


'''
import_from_statement:
    from aaa import bbb as ccc
          ^
          |
      module_name

import_statement:
    import aaa as bbb
            ^
            |
        module_name
'''
class ImportParser:
    def __init__(self, current_module_name: str):
        self.modules = {}
        self.current_module_name = current_module_name

    def get_raw_content(self) -> List[str]:
        contents = []
        for k, v in self.modules.items():
            contents.append(v)
        return contents

    def parse_one(self, node: ts.Node):
        assert node.type in ["import_statement", "import_from_statement"]

        if node.type == "import_from_statement":
            module_name_node = node.child_by_field_name("module_name")
            assert module_name_node.child_count >= 1

            module_name = ""
            dotnames = module_name_node.children
            if module_name_node.type == "relative_import":
                assert module_name_node.child_count == 2 # from ((.)(aaa)) import bbb as ccc
                dotnames = module_name_node.children[1].children
                module_name += self.current_module_name

            for dotname in dotnames:
                if dotname.text == b'.':
                    continue
                if len(module_name) == 0:
                    module_name = dotname.text.decode()
                else:
                    module_name += f"/{dotname.text.decode()}"
            # print("import from:", module_name)
            self.modules[module_name] = node.text.decode()
        elif node.type == "import_statement":
            # print(node)
            assert node.child_count == 2

            module_name_node = node.child_by_field_name("name")
            dotname_node = module_name_node
            if module_name_node.type == "aliased_import":
                assert module_name_node.child_count == 3  # import (aaa) (as) (bbb)
                assert module_name_node.children[1].text == b'as'

                dotname_node = module_name_node.children[0]
            assert dotname_node.type == "dotted_name"

            module_name = dotname_node.children[0].text.decode()
            assert module_name != '.', "import_statement doesn't allow relative import"
            for dotname in dotname_node.children[1:]:
                if dotname.text == b'.':
                    continue
                module_name += f"/{dotname.text.decode()}"
            # print("import:", module_name)
            self.modules[module_name] = node.text.decode()

class DocFileChunker:
    def __init__(self, filepath: str, current_module: str, chunk_granularity: str, parser: Parser):
        self.import_parser = ImportParser(current_module)
        self.filepath = filepath
        self.current_module = current_module
        self.max_chunk_size = 32*1024
        self.parser = parser
        self.metadata_chunks: List[str] = []
        self.metadata_chunks.extend(self.get_metadata_chunks())

        self.text_chunks: List[str] = []

        assert chunk_granularity in ["class", "function", "file"]
        self.chunk_granularity = chunk_granularity

        print(f"File \'{self.filepath}\': {self.metadata_chunks}")

    def get_metadata_chunks(self) -> List[str]:
        desc = CODE_REPO_TEMPLATE["DOC_DESCRIPTION"].format(
            module_name=self.current_module,
            filename=os.path.basename(self.filepath),
            filepath=self.filepath
        )
        return [desc]

    def parse(self) -> List[str]:
        content = ""
        with open(self.filepath, 'rb') as f:
            content = f.read()
        
        tree = self.parser.parse(content)
        root_node = tree.root_node

        n_char: int = 0
        if self.chunk_granularity == "file":
            self.text_chunks.append(self.format_text_chunk(content=content.decode()))
            n_char = len(content.decode())
        else:
            assert self.chunk_granularity == "function"
            chunks, n_char = self.chunking(root_node, content.decode(), 0)
            self.text_chunks.extend(chunks)

        print(f"File \'{self.filepath}\': {n_char} characters")
        return self.text_chunks
    
    def get_chunks(self) -> List[str]:
        chunks: List[str] = []

        chunks.extend(self.metadata_chunks)
        chunks.extend(self.text_chunks)

        print(f"File '{self.filepath}': {len(chunks)} chunks")
        return chunks

    def format_text_chunk(self, content: str) -> str:
        return CODE_REPO_TEMPLATE["DOC_CHUNK"].format(
                        filename=self.filepath,
                        module_name=self.current_module,
                        text_chunk_content=content
                    )

    def chunking(self, node: ts.Node, content: str, last_end: int = 0) -> Tuple[List[str], int]:
        chunks = []
        n_char = 0
        cur = ""
        for child in node.children:
            len_child = child.end_byte - child.start_byte
            if len_child > self.max_chunk_size:
                if len(cur) > 0:
                    chunks.append(self.format_text_chunk(content=cur))
                    n_char += len(cur)
                cur = ""
                subchunks, cnt = self.chunking(child, content, last_end)
                n_char += cnt
                chunks.extend(subchunks)
            elif len(cur) + len_child > self.max_chunk_size:
                chunks.append(self.format_text_chunk(content=cur))
                n_char += len(cur)
                cur = content[last_end : child.end_byte]
            else:
                cur += content[last_end : child.end_byte]
            last_end = child.end_byte
        if len(cur) > 0:
            chunks.append(self.format_text_chunk(content=cur))
            n_char += len(cur)
        return chunks, n_char


class CodeFileChunker:
    def __init__(self, filepath: str, current_module: str, chunk_granularity: str, parser: Parser):
        self.import_parser = ImportParser(current_module)
        self.filepath = filepath
        self.current_module = current_module
        self.max_chunk_size = 128*1024
        self.parser = parser
        self.metadata_chunks: List[str] = []
        self.metadata_chunks.extend(self.get_code_metadata_chunks())

        self.import_chunk: str = None
        self.code_chunks: List[str] = []

        assert chunk_granularity in ["class", "function", "file"]
        self.chunk_granularity = chunk_granularity

        print(f"File \'{self.filepath}\': {self.metadata_chunks}")

    def get_code_metadata_chunks(self) -> List[str]:
        desc = CODE_REPO_TEMPLATE["CODE_DESCRIPTION"].format(
            module_name=self.current_module,
            filename=os.path.basename(self.filepath),
            filepath=self.filepath
        )
        return [desc]

    def parse(self) -> List[str]:
        content = ""
        with open(self.filepath, 'rb') as f:
            content = f.read()
        
        tree = self.parser.parse(content)
        root_node = tree.root_node

        # Parse import chunks
        for child in root_node.children:
            # print(child.type)
            if child.type in ["import_from_statement", "import_statement"]:
                self.import_parser.parse_one(child)
        self.import_chunk = CODE_REPO_TEMPLATE["CODE_IMPORTS"].format(
            filename=self.filepath,
            module_name=self.current_module,
            code_chunk_content="\n".join(self.import_parser.get_raw_content())
        )

        # print(f"File \'{self.filepath}\': {self.import_chunk}")

        n_char: int = 0
        if self.chunk_granularity == "file":
            self.code_chunks.append(self.format_code_chunk(content=content.decode()))
            n_char = len(content.decode())
        elif self.chunk_granularity == "class":
            chunks, n_char = self.chunk_by_class(root_node, content.decode())
            self.code_chunks.extend(chunks)
        else:
            assert self.chunk_granularity == "function"
            chunks, n_char = self.chunking(root_node, content.decode(), 0)
            self.code_chunks.extend(chunks)

        print(f"File \'{self.filepath}\': {n_char} characters")
        return self.code_chunks
    
    def get_chunks(self) -> List[str]:
        chunks: List[str] = []

        chunks.extend(self.metadata_chunks)
        chunks.append(self.import_chunk)
        chunks.extend(self.code_chunks)

        print(f"File '{self.filepath}': {len(chunks)} chunks")

        return chunks

    def format_code_chunk(self, content: str) -> str:
        return CODE_REPO_TEMPLATE["CODE_CHUNK"].format(
                        filename=self.filepath,
                        module_name=self.current_module,
                        code_chunk_content=content
                    )

    def chunk_by_class(self, node: ts.Node, content: str) -> Tuple[List[str], int]:
        chunks = []
        n_char = 0
        cur = ""

        last_end = 0
        for child in node.children:
            len_child = child.end_byte - child.start_byte
            cur += content[last_end : child.end_byte]
            if child.type in ["class_definition", "decorated_definition"]:
                chunks.append(self.format_code_chunk(content=cur))
                n_char += len(cur)
                cur = ""
            last_end = child.end_byte
        
        if len(cur) > 0:
            chunks.append(self.format_code_chunk(content=cur))
            n_char += len(cur)
            
        return chunks, n_char

    def chunking(self, node: ts.Node, content: str, last_end: int = 0) -> Tuple[List[str], int]:
        chunks = []
        n_char = 0
        cur = ""
        for child in node.children:
            len_child = child.end_byte - child.start_byte
            if len_child > self.max_chunk_size:
                if len(cur) > 0:
                    chunks.append(self.format_code_chunk(content=cur))
                    n_char += len(cur)
                cur = ""
                subchunks, cnt = self.chunking(child, content, last_end)
                n_char += cnt
                chunks.extend(subchunks)
            elif len(cur) + len_child > self.max_chunk_size:
                chunks.append(self.format_code_chunk(content=cur))
                n_char += len(cur)
                cur = content[last_end : child.end_byte]
            else:
                cur += content[last_end : child.end_byte]
            last_end = child.end_byte
        if len(cur) > 0:
            chunks.append(self.format_code_chunk(content=cur))
            n_char += len(cur)
        return chunks, n_char


class CodeRepoParser:
    def __init__(self, repo_src: Union[str, List[str]], repo_doc: Union[str, List[str]], module_name: str, chunk_granularity: str = "function"):
        if isinstance(repo_src, str):
            repo_src = [repo_src]
        if isinstance(repo_doc, str):
            repo_doc = [repo_doc]
        self.repo_src = repo_src
        self.repo_doc = repo_doc
        self.module_name = module_name
        self.parser = Parser(PY_LANGUAGE)
        self.code_files: List[str] = []
        self.text_files: List[str] = []
        self.code_parsers: List[CodeFileChunker] = []
        self.doc_parsers: List[DocFileChunker] = []
        self.repo_chunks: List[str] = []
        self.code_chunks: List[str] = []
        self.doc_chunks: List[str] = []

        # walk through src dir
        for dir in self.repo_src:
            if not os.path.exists(dir):
                raise ValueError(f"Directory {dir} doesn't exist")
            
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith('.py'):
                        self.code_files.append(os.path.join(root, file))
        pprint(self.code_files)
        print(f"Number of code files: {len(self.code_files)}")

        # walk through doc dir
        for path in self.repo_doc:
            if not os.path.exists(path):
                raise ValueError(f"Directory {path} doesn't exist")
            
            if os.path.isfile(path):
                self.text_files.append(path)
            else:
                assert os.path.isdir(path)
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.md'):
                            self.text_files.append(os.path.join(root, file))
        pprint(self.text_files)
        print(f"Number of text files: {len(self.text_files)}")

        self.repo_chunks.extend(self.get_repo_metadata_chunks())
        print(f"Metadata chunks: {self.repo_chunks}")

        for filepath in self.code_files:
            self.code_parsers.append(CodeFileChunker(filepath, self.module_name, chunk_granularity, self.parser))
        
        for filepath in self.text_files:
            self.doc_parsers.append(DocFileChunker(filepath, self.module_name, "function", Parser(MD_LANGUAGE)))

    def get_repo_metadata_chunks(self) -> List[str]:
        desc = CODE_REPO_TEMPLATE["REPO_DESCRIPTION"].format(module_name=self.module_name, code_files=self.code_files)
        return [desc]

    def parse(self):
        for file_parser in self.code_parsers:
            file_parser.parse()
            self.code_chunks.extend(file_parser.get_chunks())

        for doc_parser in self.doc_parsers:
            doc_parser.parse()
            self.doc_chunks.extend(doc_parser.get_chunks())
        
    def get_chunks(self) -> List[str]:
        chunks = []

        chunks.extend(self.repo_chunks)
        chunks.extend(self.code_chunks)
        chunks.extend(self.doc_chunks)
        return chunks

def chunking_code_repo(repo_desc: str, overlap_token_size, max_token_size, tiktoken_model="gpt-4o"):
    repo = json.loads(repo_desc)
    repo_name = repo["name"]
    repo_src = repo["src"]
    repo_doc = repo["doc"]

    results = []
    repo_parser = CodeRepoParser(repo_src, repo_doc, repo_name, "class")
    repo_parser.parse()

    chunks = repo_parser.get_chunks()
    for index, chunk in enumerate(chunks):
        results.append({
            "content": chunk.strip(),
            "chunk_order_index": index,
        })
    return results

if __name__ == "__main__":
    parser = Parser(PY_LANGUAGE)

    content = ""
    with open(testfilename, 'rb') as f:
        content = f.read()

    repo_parser = CodeRepoParser("lightrag", "lightrag", "class")

    repo_parser.parse()

    print(f"Number of chunks: {len(repo_parser.get_chunks())}")

    # parser.parse