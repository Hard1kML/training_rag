from typing import List, Tuple, Union
from sentence_transformers import SentenceTransformer, util
import json
import os
import torch 

class encoder:
 
    def __init__(self, model_name = 'cointegrated/rubert-tiny2', use_gpu: bool=False):
       
        # Проверка: model_name не должен быть пустой строкой 
        if not model_name:
            raise ValueError('Model name cannot be empty')
        
        # Определяем устройство для вычислений 
        # 'cuda' - GPU, 'cpu' - ЦП 
        # torch.cuda.is_available() проверяет, есть ли GPU в системе 
        if use_gpu and torch.cuda.is_available(): 
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        #Загрузка модели, SentenceTransformer сам скачает модель из Hugging Face 
        try: 
            self.model = SentenceTransformer(model_name, device=self.device) 
        except Exception as e:
            raise RuntimeError(f'Failed to load model: {e}')
    def encode(self, data: Union[list[str], str]) -> torch.tensor:
       
        try: 
            #Если передана одна строка, превращаем её в список с одним элементом 
            # encode() ожидает список
            if isinstance(data, str):
                data = [data]
            #Метод encode() возвращает эмбединги 
            # convert_to_tensor сделает нампай массив в torch.tensor
            embeddings = self.model.encode(data, convert_to_tensor=True)
            return embeddings
        except Exception as e:
            raise RuntimeError(f'Encoding failed: {e}')
class RAG: 
 
    def __init__(self, encoder: encoder):
   
        #Проверяем, что encoder действительно объект класса Encoder 
    
        
        self.documents=None
        self.doc_embeddings = None
        self.encoder = encoder

    def fit(self, documents: List[str]): 
        # Проверка: документы не должны быт пустыми 
        if not documents:
            raise ValueError('Список документов не может быть пустым')
        # Сохраняем документы 
        self.documents = documents 

        try:
            # Кодируем сразу все документы 
            self.doc_embeddings = self.encoder.encode(documents)
        except Exception as e: 
            raise ValueError(f'Ошибка в кодировании документов: {e}')

    def retrieve(self, query: str, retrieval_limit: int=5, similarity_threshold: float=0.5) -> Tuple[List[int], List[str]]:
       
        # Проверка: сначала нужно вызвать fit()
        if self.documents is None:
            raise ValueError('Документы еще не были установлены. Сначала вызовите функцию fit().') 

        # Проверка: retrieval_limit от 1 до 10
        if not (1<= retrieval_limit <=10):
            raise ValueError('retrieval_limit должен быть в диапазоне от 1 до 10')

        # Проверка: Нельзя запросить больше документов чем есть 
        if retrieval_limit > len(self.documents): 
            raise ValueError(f'retrieval_limit ({retrieval_limit}) > count documents ({len(self.documents)})')

        # Проверка similarity treshold от 0 до 1 
        if not (0 <= similarity_threshold <=1):
            raise ValueError('similarity_treshold must be between 0 and 1')

        # Кодируем запрос пользователя (один текст → тензор)
        query_embedding = self.encoder.encode(query)

        # Считаем косинусную похожесть между запросом и каждым документом 
        # [0] - берем первую строку, потому что у нас один запрос
        similarities = util.cos_sim(query_embedding, self.doc_embeddings)[0]

        scores = [(i, similarities[i].item()) for i in range(len(self.documents))]

        # Сортируем по убыванию похожести 
        scores.sort(key = lambda x: x[1], reverse=True)
    
        filtered_scores = [(i, score) for i, score in scores if score >= similarity_threshold]
        
        # Берем первые retrieval_limit индексы 
        top_indices = [i for i, score in filtered_scores[:retrieval_limit]]

        if not top_indices:
            return [], []

        retrieved_docs = [self.documents[i] for i in top_indices]
        return top_indices, retrieved_docs
    def create_prompt_template(self, query: str, retrieved_docs: List[str]) -> str:
       
        prompt = "Instructions: Based on the relevant documents, generate a comprehensive response to the user's query.\n" 
        
        prompt += 'Relevant Documents:\n'

        for i, doc in enumerate(retrieved_docs): 
            prompt += f'Documents {i+1}: {doc}\n' 

        prompt += f'User query: {query}\n'

        return prompt 
    
    def _generate(self, query:str, retrieved_docs: List[str]) -> str:
        """
        Generate a response based on the retrieval documents and query.
        args: 
            query:str
            retrieval_docs: List[str]

        returns: 
            str: the generated response
        """
        pass
        
    def run(self, query:str) -> str: 
      
        indices, retrieved_docs = self.retrieve(query)

        generated_response = self._generate(query, retrieved_docs)
        
        return generated_response 

class RAGEval:
    
    def __init__(
        self,
        documents_path: str,
        questions_path: str,
        retrieval_limit: int = 5,
        similarity_threshold: float = 0.5
    ):
        
        self.documents = self.load_documents(documents_path)
        self.questions = self.load_questions(questions_path)
        self.retrieval_limit = retrieval_limit
        self.similarity_threshold = similarity_threshold

        if not self.documents:
            raise ValueError("The documents list is empty.")

        if not self.questions:
            raise ValueError("The questions list is empty.")

        self.encoder = encoder()
        self.rag = RAG(self.encoder)
        self.rag.fit(self.documents)

    def load_documents(self, path: str) -> List[str]:
        '''Загрузка json файла'''

        if not os.path.exists(path): 
            raise FileNotFoundError(f'Файл не найден: {path}')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                documents = json.load(f) 
        except json.JSONDecodeError as e:
            raise ValueError(f'Ошибка в JSON файле: {e}')

        if not isinstance(documents, list):
            raise ValueError('Ожидался список, получен: {type(documents).__name__}')
        return [self.validate_document(item) for item in documents]



    def validate_document(self, doc) -> str:
        if not isinstance(doc, dict):
            raise ValueError(f'Documets не является словарём') 
        if 'content' not in doc: 
            raise ValueError('Documents не содержит "content" ')
        return doc['content']


    def load_questions(self, path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'файл не найден: {path}')
        try: 
            with open(path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Ошибка в JSON файле: {e}')
        if not isinstance(questions, list):
            raise ValueError(f'Ожидался список, получен {type(questions).__name__}')
        return [self.validate_question(item) for item in questions]


    def validate_question(self, question) -> str:
        if not isinstance(question, dict):
            raise ValueError(f'questions не является словарём')
        if 'question' not in question:
            raise ValueError('questions не содержит поле "question"')
        return question['question']

    def evaluate(self, threshold: int = 1) -> Tuple[float, List[int], List[int]]:
        doc_usage=[0 for i in range(len(self.documents))]
        questions_wo_docs=[]
      
        for q_idx, question in enumerate(self.questions):
       
            indices, _ = self.rag.retrieve(
                query=question, 
                retrieval_limit=self.retrieval_limit, 
                similarity_threshold=self.similarity_threshold
            )
        
            if not indices:
                questions_wo_docs.append(q_idx)
            else:
                for doc_idx in indices:
                    doc_usage[doc_idx] += 1
    
       
        useless_docs = [i for i, count in enumerate(doc_usage) if count < threshold]
    
        
        total_docs = len(self.documents)
        total_questions = len(self.questions)
    
        rag_score = 1 - (len(useless_docs) / total_docs + len(questions_wo_docs) / total_questions)
    
       
        return rag_score, useless_docs, questions_wo_docs


eval = RAGEval('data/documents.json', 'data/questions.json', retrieval_limit=1, similarity_threshold=0.5)
score, useless, no_docs = eval.evaluate()
print(f'Score: {score:.2f}')
print(f'Useless docs: {useless}')
print(f'Questions without docs: {no_docs}')