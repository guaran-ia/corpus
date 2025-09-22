import unittest
from language_identifier import LanguageIdentifier
from pprint import pprint


class TestLanguageIdentifier(unittest.TestCase):

    def setUp(self):
        self.lang_identifier = LanguageIdentifier()
        self.sample_text = "La sintaxis de la formación de estos ñe'ẽnga es la siguiente: "\
                           "La primera frase se dice según la situación del momento, dentro "\
                           "de aparente normalidad. La segunda frase sin embargo, señala que "\
                           "también pudiera decir lo mismo alguien que está en una situación "\
                           "completamente diferente, o con otro sentido. Añetehápe, heta jevy "\
                           "upe ñe'ẽjoapy mokõiha reko hasy oiko haĝua térã ndaikatúi voi, ha "\
                           "upéare ijuky."
    
    def test_identify_languages_k_1(self):
        result = self.lang_identifier.identify_languages(self.sample_text, k=1)
        pprint(result)
        self.assertIn('source', result)
        self.assertIn('voting', result)
        self.assertIn('languages', result)
        self.assertEqual('grn', result['languages'][0])  # Expecting Guarani as the top language
        self.assertEqual('glotlid', result['source'])
        self.assertEqual('all_agree', result['voting'])
        
    def test_identify_languages_k_2(self):
        result = self.lang_identifier.identify_languages(self.sample_text, k=2)
        pprint(result)
        self.assertIn('source', result)
        self.assertIn('voting', result)
        self.assertIn('languages', result)
        self.assertEqual('grn', result['languages'][0][0])  # Expecting Guarani as the top language
        self.assertEqual('spa', result['languages'][1][0])  # Expecting Spanish as the second language
        self.assertEqual('glotlid', result['source'])
        self.assertEqual('not_applicable_k_greater_than_1', result['voting'])

    def test_identify_languages_raw_output(self):
        result = self.lang_identifier.identify_languages(self.sample_text, raw_output=True)
        pprint(result)
        self.assertIn('glotlid', result)
        self.assertIn('fasttext', result)
        self.assertIn('openlid', result)

    def test_identify_languages_invalid_k(self):
        with self.assertRaises(ValueError):
            self.lang_identifier.identify_languages(self.sample_text, k=0)

if __name__ == '__main__':
    unittest.main()