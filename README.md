# code-generator-app

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This is a simple project that allows users to create their own tools and **ReAct** agent to generate code. Here are the main features of this project:
* Create tools that extract information from different types of files
* Create **ReAct** agent from a selection of tools
* Generate code files with your agent and suitable prompts
* Download code files generated by agent

### Built With
* [![Ollama][Ollama-logo]][Ollama-url]
* [![LlamaIndex][LlamaIndex-logo]][LlamaIndex-url]
* [![Streamlit][Streamlit-logo]][Streamlit-url]

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* Install Ollama: follow the guide on this [link][Ollama-url].

### Installation
1. Clone the repository from Github:
    ```sh
    git clone https://github.com/MrNobody65/code-generator-app.git
    ```
2. Create and activate python virtual environment with `conda`
    ```sh
    conda create -p venv python==3.11.0
    conda activate ./venv
    ```
3. Install dependencies
    ```sh
    pip install -r requirements.txt
    ```

4. Create `.env` file based on `.env.example` and enter your API key from [LlamaCloud](https://cloud.llamaindex.ai/api-key)

5. Run the application
    ```sh
    streamlit run app.py
    ```

<!-- MARKDOWN LINKS & IMAGES -->
[Ollama-logo]: https://img.shields.io/badge/Ollama-FFFFFF?style=for-the-badge
[Ollama-url]: https://github.com/ollama/ollama
[LlamaIndex-logo]: https://img.shields.io/badge/LlamaIndex-270D57?style=for-the-badge
[LlamaIndex-url]: https://github.com/run-llama/llama_index
[Streamlit-logo]: https://img.shields.io/badge/Streamlit-%23FF4B4B?style=for-the-badge&logo=streamlit&logoColor=FFFFFF
[Streamlit-url]: https://github.com/streamlit/streamlit