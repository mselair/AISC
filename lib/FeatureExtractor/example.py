from FeatureExtractor import SleepSpectralFeatureExtractor

fs = 500 # vzorkovacka
segm_size = 30 # delka segmentu ze ktereho extrahujes featury v sekundach
fbands = [[1, 4],
 [4, 8],
 [8, 12],
 [12, 14],
 [14, 20],
 [20, 30]] # nadefinujes si frekvencni bandy

f = 10
a = 0
t = np.arange(0, 1000, 1/fs)
x = a + np.sin(2*np.pi*f*t)


Extractor = SleepSpectralFeatureExtractor() # nainicializujes
feature_values, feature_names = Extractor(x=[x], fs=fs, segm_size=segm_size, fbands=fbands, n_processes=2) # pustis to tam, vstup x muze byt np.array, popr list obsahujicic vice signalu - np arrays
# na vystupu dostanes 2 promenne viz vyse
# pokud chces zmenit featury ukazka vFeatureExtractor/__init__.py -> method __call__ -> self.__extraction_functions
# self._extraction_functions = [self.normalized_entropy, self.MeanFreq, self.MedFreq, self.mean_bands, self.rel_bands, self.normalized_entropy_bands]
# ted jsou tam vsechny featury, ktere to umi spocitat, muzes tam nechat jen kteere chces, pokud chces delta - beta ratio, ktere je asi nejlepsi, tak akorat podelis ty relativni bandy o zvolenych frekvencich
# pokud to budes pocitat na vic kanalech, tak tam posli vic kanalu a pak udelej median pres kanaly.