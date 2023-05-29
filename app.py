import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import AtlasDB
from langchain.document_loaders import GitbookLoader
import os
import time
import base64

load_dotenv()

with st.sidebar:
    st.title("Langchain Atlas Gitbook")
    st.markdown(
        """
    ## About
    This app is an LLM-powered Gitbook chatbot built using:
    - [Streamlit](https://streamlit.io/) (for the UI)
    - [LangChain](https://python.langchain.com/) (for the LLM)
    - [OpenAI](https://platform.openai.com/docs/models) (for the LLM)
    - [AtlasDB](https://atlas.nomic.ai/) (for the vector store and visualizations)
    """
    )


def main():
    st.header("Ask Gitbook docs  💬 and visualize embeddings")

    open_api_key = st.text_input("Enter your OpenAI API Key:")
    if open_api_key:
        os.environ["OPENAI_API_KEY"] = open_api_key
        st.info("OpenAI API Key set successfully")
    else:
        st.error("Please enter your OpenAI API Key", icon="🚨")
        return

    gitbook_url = st.text_input("Enter your Gitbook Docs URL(can be custom domain):")

    if gitbook_url:
        loader = GitbookLoader(gitbook_url, load_all_paths=True)
        with st.spinner("Loading the data and creating embeddings, please wait..."):
            all_pages_data = loader.load()

            text = ""
            for document in all_pages_data:
                text += document.page_content

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100, length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # base64 encoded string of the gitbook url
            store_name = base64.b64encode(gitbook_url.encode("utf-8")).decode("utf-8")
            store_name = store_name[:-2]
            st.write(store_name)

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                st.write("Embeddings Loaded from the Disk")
                with st.expander("Visualize embeddings"):
                    st.write(VectorStore.project.maps[0])
            else:
                VectorStore = AtlasDB.from_texts(
                    texts=chunks,
                    embeddings=OpenAIEmbeddings(),
                    name=store_name,
                    description="test_index",
                    api_key=os.getenv("NOMIC_API_KEY"),
                    index_kwargs={"build_topic_model": True},
                    embedding=OpenAIEmbeddings(),
                )
                VectorStore.project.wait_for_project_lock()
                time.sleep(5)
                st.write("Embeddings Created and Saved to Disk")
                with st.expander("Visualize embeddings"):
                    st.write(VectorStore.project.maps[0])
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # Accept user questions/query
            query = st.text_input("Ask questions about the gitbook docs:")

        if query:
            with st.spinner("Fetching the answer..."):
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
