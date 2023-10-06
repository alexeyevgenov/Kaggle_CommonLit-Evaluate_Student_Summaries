import re
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
import spacy
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk import sent_tokenize, pos_tag
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from spellchecker import SpellChecker
import ast
from feature_generation.config import CONFIG
import textstat
import Levenshtein


N_ROWS = 50
features_for_norm = ["summary_length", "spelling_err_num", "word_overlap_count", "bigram_overlap_count",
                     "trigram_overlap_count", "quotes_count"]
drop_columns = ["prompt_id", "prompt_question", "prompt_title", "prompt_text", "student_id", "text", "full_text",
                "embeddings"]


# nltk.download('stopwords')  # todo: must to be downloaded
# nltk.download('punkt')  # todo: must to be downloaded
# python - m spacy download en_core_web_sm  # todo: must to be downloaded
# nltk.download('averaged_perceptron_tagger')  # todo: must to be downloaded


class Preprocessor:
    def __init__(self, test_mode: bool) -> None:
        self.test_mode = test_mode
        self.STOP_WORDS = set(stopwords.words('english'))
        self.spacy_ner_model = spacy.load('en_core_web_sm')
        self.speller = Speller(lang='en')
        self.spellchecker = SpellChecker()
        # self.scaler = StandardScaler()

    @staticmethod
    def calculate_text_similarity(row):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([row['prompt_text'], row['text']])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]

    @staticmethod
    def sentiment_analysis(text):
        analysis = TextBlob(text)   # .sentiment
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity

    @staticmethod
    def sentiment_analysis_modin(text):
        analysis = TextBlob(text)
        return analysis.sentiment

    @staticmethod
    def calculate_sentiment_vader(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score

    @staticmethod
    def calculate_lexical_characteristics(text):
        words = word_tokenize(text.lower())

        # Рассчитываем частоту встречаемости каждого слова
        word_freq = FreqDist(words)
        # Преобразуем частоты в вероятности
        word_prob = [word_freq[word] / len(words) for word in word_freq.keys()]
        # Рассчитываем лексическую энтропию
        lexical_entropy = scipy_entropy(word_prob, base=2)
        # Рассчитываем разнообразие лексики
        lexical_diversity = len(word_freq) / len(words)

        return lexical_entropy, lexical_diversity

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """

        def check_is_stop_word(word):
            return word in self.STOP_WORDS

        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    @staticmethod
    def ngrams(token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)

    @staticmethod
    def quotes_count(row):  # todo: it is good to add a function of checking for copy-pasting
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary) > 0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):
        wordlist = text.split()
        amount_miss = len(list(self.spellchecker.unknown(wordlist)))
        return amount_miss

    @staticmethod
    def calculate_unique_words(text):
        unique_words = set(text.split())
        return len(unique_words)

    def add_spelling_dictionary(self, tokens: List[str]) -> None:  # List[str]
        """dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens)
        self.speller.nlp_data.update({token: 1000 for token in tokens})

    @staticmethod
    def calculate_pos_ratios(text):
        pos_tags = pos_tag(word_tokenize(text))
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_words = len(pos_tags)
        ratios = {tag: count / total_words for tag, count in pos_counts.items()}
        return ratios

    @staticmethod
    def calculate_punctuation_ratios(text):
        total_chars = len(text)
        punctuation_counts = Counter(char for char in text if char in '.,!?;:"()[]{}')
        ratios = {char: count / total_chars for char, count in punctuation_counts.items()}
        return ratios

    @staticmethod
    def calculate_keyword_density(row):
        keywords = set(row['prompt_text'].split())
        text_words = row['text'].split()
        keyword_count = sum(1 for word in text_words if word in keywords)
        return keyword_count / len(text_words)

    @staticmethod
    def calculate_sentiment_scores(text):
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        return sentiment_scores

    def run_modin(self) -> None:
        import modin.pandas as pd
        import ray
        ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})

        for i in range(CONFIG.num_folds):
            print(f"\nPREPROCESSING THE FOLD {i}:")

            input_df = pd.read_feather(path=CONFIG.init_data_storage + f"/fold {i}.ftr")
            if self.test_mode:
                input_df = input_df[:N_ROWS]

            # feature generation with help of textstat library
            input_df["flesch_reading_ease"] = input_df["text"].apply(lambda x: textstat.flesch_reading_ease(x))
            input_df["flesch_kincaid_grade"] = input_df["text"].apply(lambda x: textstat.flesch_kincaid_grade(x))
            input_df["gunning_fog"] = input_df["text"].apply(lambda x: textstat.gunning_fog(x))
            input_df["smog_index"] = input_df["text"].apply(lambda x: textstat.smog_index(x))
            input_df["automated_readability_index"] = input_df["text"].apply(
                lambda x: textstat.automated_readability_index(x))
            input_df["coleman_liau_index"] = input_df["text"].apply(lambda x: textstat.coleman_liau_index(x))
            input_df["linsear_write_formula"] = input_df["text"].apply(lambda x: textstat.linsear_write_formula(x))
            input_df["dale_chall_readability_score"] = input_df["text"].apply(
                lambda x: textstat.dale_chall_readability_score(x))
            input_df["text_standard"] = input_df["text"].apply(lambda x: textstat.text_standard(x, float_output=True))
            input_df["spache_readability"] = input_df["text"].apply(lambda x: textstat.spache_readability(x))
            input_df["mcalpine_eflaw"] = input_df["text"].apply(lambda x: textstat.mcalpine_eflaw(x))
            input_df["reading_time"] = input_df["text"].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
            input_df["syllable_count"] = input_df["text"].apply(lambda x: textstat.syllable_count(x))
            input_df["lexicon_count"] = input_df["text"].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
            input_df["sentence_count"] = input_df["text"].apply(lambda x: textstat.sentence_count(x))
            input_df["char_count"] = input_df["text"].apply(lambda x: textstat.char_count(x, ignore_spaces=True))
            input_df["letter_count"] = input_df["text"].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
            input_df["polysyllabcount"] = input_df["text"].apply(lambda x: textstat.polysyllabcount(x))
            input_df["monosyllabcount"] = input_df["text"].apply(lambda x: textstat.monosyllabcount(x))

            # Levenshtein distance
            input_df['levenshtein_text_to_promt'] = input_df.apply(
                lambda x: Levenshtein.distance(x['prompt_text'], x['text']), axis=1)
            input_df['levenshtein_text_to_promt_norm'] = input_df.apply(
                lambda x: x['levenshtein_text_to_promt'] / len(x['prompt_text']), axis=1)

            # lexical_entropy, lexical_diversity
            input_df['lexical_entropy'], input_df['lexical_diversity'] = zip(
                *input_df['text'].apply(self.calculate_lexical_characteristics))

            # the length of a text
            input_df['text_len_symbols'] = input_df['text'].apply(len)
            input_df['text_len_words'] = input_df['text'].apply(lambda text: len(text.split()))
            input_df['text_len_sentences'] = input_df['text'].apply(lambda text: len(sent_tokenize(text)))

            # the length of words and sentences
            input_df['sentence_length_in_words'] = input_df['text_len_words'] / input_df['text_len_sentences']
            input_df['word_length_in_symbols'] = input_df['text_len_symbols'] / input_df['text_len_words']

            # Sentiment analysis
            # input_df['vader_sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_vader)
            # input_df['vader_sentiment_positive'] = input_df['vader_sentiment_scores'].apply(lambda x: x['pos'])
            # input_df['vader_sentiment_negative'] = input_df['vader_sentiment_scores'].apply(lambda x: x['neg'])
            # input_df['vader_sentiment_neutral'] = input_df['vader_sentiment_scores'].apply(lambda x: x['neu'])
            # input_df['vader_sentiment_compound'] = input_df['vader_sentiment_scores'].apply(lambda x: x['compound'])

            input_df["prompt_length"] = input_df["prompt_text"].apply(lambda x: len(word_tokenize(x)))
            input_df["prompt_tokens"] = input_df["prompt_text"].apply(word_tokenize)

            input_df["summary_length"] = input_df["text"].apply(lambda x: len(word_tokenize(x)))
            input_df["summary_tokens"] = input_df["text"].apply(word_tokenize)

            # Add prompt tokens into spelling checker dictionary
            input_df["prompt_tokens"].apply(self.add_spelling_dictionary)

            # fix misspelling
            # input_df["fixed_summary_text"] = input_df["text"].apply(self.speller)

            # count misspelling
            input_df["spelling_err_num"] = input_df["text"].apply(self.spelling)

            input_df['word_count'] = input_df['text'].apply(lambda x: len(x.split()))
            input_df['sentence_length'] = input_df['text'].apply(lambda x: len(x.split('.')))
            input_df['vocabulary_richness'] = input_df['text'].apply(lambda x: len(set(x.split())))

            input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
            input_df['num_unq_words'] = [len(list(set(x.lower().split(' ')))) for x in input_df.text]
            input_df['num_chars'] = [len(x) for x in input_df.text]

            # Additional features
            input_df['avg_word_length'] = input_df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
            input_df['comma_count'] = input_df['text'].apply(lambda x: x.count(','))
            input_df['semicolon_count'] = input_df['text'].apply(lambda x: x.count(';'))

            # after merge preprocess
            input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

            input_df['word_overlap_count'] = input_df.apply(self.word_overlap_count, axis=1)

            input_df['bigram_overlap_count'] = input_df.apply(self.ngram_co_occurrence, args=(2,), axis=1)
            input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

            input_df['trigram_overlap_count'] = input_df.apply(self.ngram_co_occurrence, args=(3,), axis=1)
            input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

            input_df['quotes_count'] = input_df.apply(self.quotes_count, axis=1)

            input_df['question_count'] = input_df['text'].apply(lambda x: x.count('?'))
            input_df['pos_ratios'] = input_df['text'].apply(self.calculate_pos_ratios)

            # Convert the dictionary of POS ratios into a single value (mean)
            input_df['pos_mean'] = input_df['pos_ratios'].apply(lambda x: np.mean(list(x.values())))
            input_df['punctuation_ratios'] = input_df['text'].apply(self.calculate_punctuation_ratios)

            # Convert the dictionary of punctuation ratios into a single value (sum)
            input_df['punctuation_sum'] = input_df['punctuation_ratios'].apply(lambda x: np.sum(list(x.values())))
            try:
                input_df['keyword_density'] = input_df.apply(self.calculate_keyword_density, axis=1)
                input_df['jaccard_similarity'] = input_df.apply(
                    lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(
                        set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)
            except Exception as ex:
                print(ex)

            analysis_sentiment = input_df['text'].apply(lambda x: pd.Series(self.sentiment_analysis_modin(x)))
            input_df['sentiment_polarity'] = analysis_sentiment.apply(lambda x: x[0])
            input_df['sentiment_subjectivity'] = analysis_sentiment.apply(lambda x: x[1])

            input_df['text_similarity'] = input_df.apply(self.calculate_text_similarity, axis=1)

            # Calculate sentiment scores for each row
            input_df['sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_scores)

            # Convert sentiment_scores into individual columns
            sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
            input_df = pd.concat([input_df, sentiment_columns], axis=1)
            input_df['sentiment_scores_prompt'] = input_df['prompt_text'].apply(self.calculate_sentiment_scores)

            # Convert sentiment_scores_prompt into individual columns
            sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
            sentiment_columns_prompt.columns = [col + '_prompt' for col in sentiment_columns_prompt.columns]
            input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)

            # embeddings preparation
            input_df.rename(columns={"embeddings": "stringed_embeddings"}, inplace=True)
            input_df["embeddings"] = input_df["stringed_embeddings"].apply(lambda x: ast.literal_eval(x))
            embeddings_length = len(input_df["embeddings"][0])
            embedding_columns = pd.DataFrame(input_df['embeddings'].to_list(),
                                             columns=[f"emb_{i}" for i in range(embeddings_length)])
            input_df = pd.concat([input_df, embedding_columns], axis=1)

            input_df = input_df.drop(columns=["summary_tokens", "prompt_tokens", "stringed_embeddings",
                                              "pos_ratios", "punctuation_ratios", "prompt_length",
                                              "sentiment_scores", "sentiment_scores_prompt", "neg_prompt",
                                              "neu_prompt", "pos_prompt", "compound_prompt"])
            input_df = input_df.drop(columns=drop_columns)
            # Store to feather
            input_df.to_feather(CONFIG.storage + "/" + CONFIG.version + f"/preprocessed fold {i}.ftr")


def group_folds_in_a_single_df(path: str, num_folds: int) -> pd.DataFrame:
    """
    Downloads data from 'preprocessed fold xxx.ftr' with a feature engineered data +
    separately placed by columns embeddings ,
    """
    total_df = pd.DataFrame()
    for i in range(num_folds):
        fold_df = pd.read_feather(path + f"/preprocessed fold {i}.ftr")
        fold_df["fold"] = i
        if total_df.empty:
            total_df = fold_df
        else:
            total_df = pd.concat([total_df, fold_df], axis=0, ignore_index=True)
    return total_df


def split_data_on_train_test(data: pd.DataFrame, fold_nummer: int, target_name: str) -> (
        tuple)[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = data[data["fold"] != fold_nummer].drop(columns=["fold"] + CONFIG.data.targets)
    y_train = data[data["fold"] != fold_nummer][target_name]

    x_test = data[data["fold"] == fold_nummer].drop(columns=["fold"] + CONFIG.data.targets)
    y_test = data[data["fold"] == fold_nummer][target_name]
    return x_train, x_test, y_train, y_test


def normalize_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df[train_df.columns] = scaler.fit_transform(train_df)
    test_df[test_df.columns] = scaler.transform(test_df)
    return train_df, test_df, scaler


def remove_highly_collinear_variables(df: pd.DataFrame, collinearity_threshold: float) -> pd.DataFrame:
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > collinearity_threshold)]
    print(f"There are defined {len(to_drop)} features with correlation greater than {collinearity_threshold}: {to_drop}"
          )
    df.drop(to_drop, axis=1, inplace=True)
    return df


def clean_lexile(lexile):
    """
    Function to clean lexile feature
    Args:
        lexile (str or float): The lexile measure as a string or a float

    Returns:
        int or np.nan: The cleaned lexile measure as an integer, or np.nan for 'Non-Prose' or 'nan' values
    """
    if pd.isnull(lexile):
        return np.nan
    elif isinstance(lexile, str):
        if lexile == 'Non-Prose':
            return np.nan
        else:
            # Remove the 'L' at the end and convert to integer
            return int(lexile.rstrip('L'))
    else:
        # If lexile is a float (or any non-string data type), convert to int and return
        return int(lexile)


def classify_author(spacy_ner_model, author):
    # Process the text
    doc = spacy_ner_model(author)

    # Check if any of the entities are labeled as 'PERSON'
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return 'person'

    # If no 'PERSON' entity is found, return 'org'
    return 'org'


def encode_author_type(df):
    """
    Function to encode author_type feature
    Args:
        df (pd.DataFrame): The DataFrame with 'author_type' column

    Returns:
        pd.DataFrame: The DataFrame with 'author_type' replaced with numerical values
    """
    le = LabelEncoder()
    df['author_type'] = le.fit_transform(df['author_type'])
    return df


def clean_grade(df):
    """
    Function to clean grade feature
    Args:
        df (pd.DataFrame): The DataFrame with 'grade' column

    Returns:
        pd.DataFrame: The DataFrame with 'grade' replaced with integer values
    """
    df['grade'] = df['grade'].astype(str).str.replace('rd Grade', '')
    df['grade'] = df['grade'].str.replace('th Grade', '')
    df['grade'] = df['grade'].apply(lambda x: int(x) if x.isdigit() else 0)
    return df


def group_and_encode_genre(df):
    """
    Function to group and encode genre feature
    Args:
        df (pd.DataFrame): The DataFrame with 'genre' column

    Returns:
        pd.DataFrame: The DataFrame with 'genre' replaced with grouped and encoded values
    """
    genre_map = {
        'Fiction': ['Poem', 'Short Story', 'Folktale', 'Fantasy', 'Science Fiction', 'Allegory',
                    'Fiction - General', 'Fable', 'Myth', 'Historical Fiction', 'Magical Realism', 'Drama'],
        'Non-Fiction': ['Informational Text', 'Non-Fiction - General', 'Biography', 'Essay', 'Memoir', 'Interview',
                        'Psychology', 'Primary Source Document', 'Autobiography'],
        'News & Opinion': ['News', 'Opinion'],
        'Historic & Legal': ['Historical Document', 'Legal Document', 'Letter'],
        'Philosophy & Religion': ['Speech', 'Religious Text', 'Satire', 'Political Theory', 'Philosophy']
    }

    # Reverse the genre_map dictionary for mapping
    reverse_genre_map = {genre: key for key, values in genre_map.items() for genre in values}

    df['genre_big_group'] = df['genre'].map(reverse_genre_map)

    # If the genre is not found in the map, assign it to 'Other'
    df['genre_big_group'] = df['genre_big_group'].fillna('Other')

    le = LabelEncoder()
    df['genre_big_group_encode'] = le.fit_transform(df['genre_big_group'])

    return df


def preprocess_and_join(df1, df2, df1_title_col, df2_title_col, grade_col):
    # Copy dataframes to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()

    # Preprocess titles
    df1[df1_title_col] = df1[df1_title_col].str.replace('"', '').str.strip()
    df2[df2_title_col] = df2[df2_title_col].str.replace('"', '').str.strip()

    # Remove duplicate grades
    df2 = df2.drop_duplicates(subset=df2_title_col, keep='first')

    # Join dataframes
    merged_df = df1.merge(df2, how='left', left_on=df1_title_col, right_on=df2_title_col)

    # Postprocess grades
    merged_df[grade_col] = merged_df[grade_col].fillna(0)
    merged_df[grade_col] = merged_df[grade_col].astype(int).astype('category')

    return merged_df
