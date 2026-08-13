from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data=TextLoader("C:\\Users\\008257\\Desktop\\RAGG Project\\notes.txt")
docs= data.load()
splitter= CharacterTextSplitter(
separator="\n"
,chunk_size=10,chunk_overlap=1)

chunks=splitter.split_documents(docs)

print(chunks[0])

# for i in chunks:
#     print(i.page_content)


