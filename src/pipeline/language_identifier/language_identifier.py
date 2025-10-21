import fasttext
import os
import pandas as pd

from huggingface_hub import hf_hub_download


class LanguageIdentifier:
    """
    A class for identifying the language of a given text using three different 
    language identification models: GLotLID, FastText, and OpenLID.
    """
    glotlid_model, fasttext_model, openlid_model = None, None, None
    models_dir_path = None
    
    def __init__(self, glotlid=True, fasttext=True, openlid=True, verbose=False):
        """
        Initializes the LanguageIdentifier class.

        Args:
            glotlid (bool, optional): Whether to load the GLotLID model. Defaults to True.
            fasttext (bool, optional): Whether to load the FastText model. Defaults to True.
            openlid (bool, optional): Whether to load the OpenLID model. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        # Get the directory of the current file
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir_path = os.path.join(file_dir, 'models')
        if glotlid:
            self.glotlid_model = self.load_glotlid_model(verbose=verbose)
        if fasttext:
            self.fasttext_model = self.load_fasttext_model(verbose=verbose)
        if openlid:
            self.openlid_model = self.load_openlid_model(verbose=verbose)
        # Construct the absolute path to the CSV file
        iso_file_path = os.path.join(file_dir, 'res', 'iso6393_macro.csv')
        self.df_isocodes = pd.read_csv(iso_file_path)

    def load_glotlid_model(self, verbose=False):
        """
        Loads the GLotLID language identification model.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            fasttext.FastText._FastText: The loaded GLotLID model, or None if an error occurred.
        """
        model_path = hf_hub_download(
            repo_id='cis-lmu/glotlid', 
            filename='model.bin', 
            cache_dir=self.models_dir_path,
            local_files_only=False
        )
        try:
            if verbose: print(f'Loading GlotLID model from {model_path}')
            return fasttext.load_model(model_path)
        except Exception as e:
            print(f'Error loading GlotLID model: {e}')
            return None

    def load_fasttext_model(self, verbose=False):
        """
        Loads the FastText language identification model.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            fasttext.FastText._FastText: The loaded FastText model, or None if an error occurred.
        """
        model_path = hf_hub_download(
            repo_id='facebook/fasttext-language-identification', 
            filename='model.bin',
            cache_dir=self.models_dir_path,
            local_files_only=False
        )
        try:
            if verbose: print(f'Loading FastText model from {model_path}')
            return fasttext.load_model(model_path)
        except Exception as e:
            print(f'Error loading FastText model: {e}')
            return None

    def load_openlid_model(self, verbose=False):
        """
        Loads the OpenLID language identification model.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            fasttext.FastText._FastText: The loaded OpenLID model, or None if an error occurred.
        """
        model_path = os.path.join(self.models_dir_path, 'lid201-model.bin')
        try:
            if verbose: print(f'Loading OpenLID model from {model_path}')
            return fasttext.load_model(model_path)
        except Exception as e:
            print(f'Error loading OpenLID model: {e}')
            return None

    def normalize_lang_code(self, lang_code):
        """
        Normalizes a language code by mapping it to its corresponding macro 
        language code using the ISO 639-3 standard.

        Args:
            lang_code (str): The language code to normalize.

        Returns:
            str: The normalized language code (macro language code if available, 
            otherwise the original code).
        """
        m_code = self.df_isocodes.loc[self.df_isocodes['I_Id'] == lang_code, 'M_Id'].values
        if m_code.size > 0:
            return m_code[0]
        else:
            return lang_code

    def process_lang_prediction(self, prediction):
        """
        Processes the raw prediction output from a language identification model.

        Args:
            prediction (tuple): The raw prediction output from a language 
            identification model (list of labels, list of confidences).

        Returns:
            list: A sorted list of tuples, where each tuple contains a language 
                  code and its confidence score, sorted in descending order of 
                  confidence. Returns None if the prediction is empty or invalid.
        """
        preds = {}
        if prediction and len(prediction[0]) > 0:
            for i in range(len(prediction[0])):
                lang_code = self.normalize_lang_code(prediction[0][i].replace('__label__', '').split('_')[0])
                confidence = prediction[1][i]
                preds[lang_code] = confidence
            return sorted(preds.items(), key=lambda x: x[1], reverse=True)
        else:
            return None
    
    def compute_prediction_result(self, identification_results, k):
        """
        Computes a final language prediction result based on the 
        predictions of the three language identification models.

        Args:
            identification_results (dict): A dictionary containing the prediction 
                                           results from each model. The keys are 
                                           'glotlid', 'fasttext', and 'openlid', 
                                           and the values are lists of
                                           (language code, confidence score) tuples, 
                                           sorted by confidence in descending order.
            k (int): The number of top language predictions to consider from each model.

        Returns:
            dict: A dictionary containing the final language prediction result. 
                  The dictionary contains the following keys:
                - 'languages': A list of (language code, confidence score) tuples, 
                  representing the distribution of predicted languages (if the models disagree).
                - 'source': The source of the prediction (e.g., 'glotlid', 'fasttext', 
                  'openlid', or 'glotlid_fasttext_openlid').
                - 'voting': A string indicating how the final prediction was 
                   determined (e.g., 'all_agree', 'agree_glotlib_fasttext',
                  'inconclusive').
        """
        if k == 1:
            glotlib_lang, fasttext_lang, openlid_lang = None, None, None
            if identification_results['glotlid'] is not None:
                glotlib_lang = identification_results['glotlid'][0][0]
            if identification_results['fasttext'] is not None:
                fasttext_lang = identification_results['fasttext'][0][0]
            if identification_results['openlid'] is not None:
                openlid_lang = identification_results['openlid'][0][0]
            # implement majority voting
            if glotlib_lang is None and \
                fasttext_lang is None and \
                openlid_lang is None:
                return None
            else:
                if glotlib_lang == fasttext_lang or glotlib_lang == openlid_lang:
                    # three models agree on the same language with high confidence,
                    # return GlotLID result
                    return {
                        'languages': identification_results['glotlid'][0],
                        'source': 'glotlid',
                        'voting': 'agree_glotlib_fasttext_openlid'
                    }
                elif glotlib_lang is not None and \
                        fasttext_lang is not None and \
                        openlid_lang is not None and \
                        fasttext_lang != openlid_lang and \
                        glotlib_lang != openlid_lang and \
                        glotlib_lang != fasttext_lang:
                    # all three models disagree and are not None, return GlotLID result
                    return {
                        'languages': identification_results['glotlid'][0],
                        'source': 'glotlid',
                        'voting': 'all_different' 
                    }
                else:
                    # two models agree on the same language with high confidence
                    if glotlib_lang is not None and \
                        fasttext_lang is not None and \
                        glotlib_lang == fasttext_lang:
                        # between fasttext and glotlib, return glotlib result
                        return {
                            'languages': identification_results['glotlib'][0],
                            'source': 'glotlib',
                            'voting': 'agree_glotlib_fasttext'
                        }
                    elif glotlib_lang is not None and \
                            openlid_lang is not None and \
                            glotlib_lang == openlid_lang:
                        # between openlid and glotlib, return glotlib result
                        return {
                            'languages': identification_results['glotlib'][0],
                            'source': 'glotlib',
                            'voting': 'agree_glotlib_openlid'
                        }
                    elif fasttext_lang is not None and \
                            openlid_lang is not None and \
                            fasttext_lang == openlid_lang:
                        # between openlid and fasttext, return openlid result
                        return {
                            'languages': identification_results['openlid'][0],
                            'source': 'openlid',
                            'voting': 'agree_fasttext_openlid'
                        }
                    else:
                        # as a fallback, return raw results
                        return {
                            'languages': identification_results,
                            'source': 'glotlid_fasttext_openlid',
                            'voting': 'inconclusive'
                        }
        else:
            # if k > 1, return GlotLID results
            return {
                'languages': identification_results['glotlid'],
                'source': 'glotlid',
                'voting': 'not_applicable_k_greater_than_1'
            }

    def identify_languages(self, text, k=3, raw_output=False):
        """
        Identifies the language of the input text using three different language 
        identification models: GLotLID, FastText, and OpenLID.

        Args:
            text (str): The text to identify the language of.
            k (int, optional): The number of top languages to return for each model. 
            Defaults to 3.
            raw_output (bool, optional): Whether to return the raw output 
            from each model. Defaults to False.

        Returns:
            dict: A dictionary containing the language identification results.
                  If raw_output is False, the dictionary contains the same format
                  as the output of compute_prediction_result()
                  If raw_output is True, the dictionary contains the raw output 
                  from each model, with the keys 'glotlid', 'fasttext', and 'openlid'.
        Raises:
            ValueError: If k is not greater than 0.
        """
        
        if k > 0:
            identification_results = {
                'glotlid': None,
                'fasttext': None,
                'openlid': None
            }
            # Identify language using GLotLID
            if self.glotlid_model:
                glotlid_prediction = self.glotlid_model.predict(text, k=k)
                identification_results['glotlid'] = self.process_lang_prediction(glotlid_prediction)
            # Identify language using FastText
            if self.fasttext_model:
                fasttext_prediction = self.fasttext_model.predict(text, k=k)
                identification_results['fasttext'] = self.process_lang_prediction(fasttext_prediction)
            # Identify language using OpenLID
            if self.openlid_model:
                openlid_prediction = self.openlid_model.predict(text, k=k)
                identification_results['openlid'] = self.process_lang_prediction(openlid_prediction)
            if raw_output:
                return identification_results 
            else:
                return self.compute_prediction_result(identification_results, k)
        else:
            raise ValueError("k must be greater than 0")
