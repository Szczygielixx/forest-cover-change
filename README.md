# Analiza zmian pokrywy leśnej

Implementacja eksperymentów pracy magisterskiej: kompozyty Landsat 8, klasyfikacja las/nie-las metodami NDVI, Random Forest, SVM RBF i U-Net oraz porównanie map z różnych lat.

## Zakres

- Landsat 8, HLS L30 v2.0, rozdzielczość 30 m, obserwacje od 1 czerwca do 30 września w latach 2015, 2020 i 2025.
- 12 obszarów w stanach Rondônia i Pará: 6 uczących, 2 walidacyjne i 4 testowe. Geometrie i podział zapisano w `configs/areas.json`, identyfikatory scen w `configs/scenes.csv`.
- 12 obrazów uczących, 4 walidacyjne i 12 testowych. Rok 2025 występuje wyłącznie w teście.
- Wspólne 60 000 etykiet dla RF, SVM i U-Net: po 5000 na obraz, po połowie każdej klasy. Próg NDVI dobierany jest na walidacji.
- Etykiety i ocena pochodzą z MapBiomas Brasil Collection 11. Wyniki oznaczają zgodność z tym produktem na odrębnych obszarach, a nie niezależny pomiar terenowy.

## Zawartość

| Katalog | Zawartość |
|---|---|
| `preprocessing/` | Wybór obszarów i scen, maskowanie, kompozyty, etykiety i wspólna próba ucząca. |
| `training/` | Klasyfikatory, strojenie, uczenie i generowanie map. |
| `evaluation/` | Macierze pomyłek, miary jakości, powierzchnie i rozbieżności przy granicach klas. |
| `experiments/` | Dodatkowe próby 120/240 tys. etykiet i wydłużone uczenie U-Net. |
| `configs/` | Obszary, źródła danych i konfiguracje użyte w badaniu. |
| `results/` | Niewielkie zestawienia wyników opisanych w pracy. |
| `tests/` | Test przepływu na sztucznych danych, bez pobierania zdjęć. |

Repozytorium nie zawiera danych rastrowych, wytrenowanych wag ani logów. Przy uruchomieniu kod zapisuje modele, predykcje, wyniki walidacji i stan uczenia w wybranym katalogu roboczym. Są one potrzebne do wyboru najlepszych wag, oceny i analizy długości uczenia.

## Szybki test bez danych i GPU

Polecenia wykonuje się z katalogu głównego repozytorium. Przykłady aktywacji środowiska poniżej dotyczą Linuxa. W Windows odpowiednikiem jest `.venv-train\Scripts\Activate.ps1`.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Testy obejmują maski jakości, kompozyt, etykiety, zagnieżdżone próby, macierze pomyłek, próg NDVI, zapis i odczyt RF oraz krótki trening i wznowienie U-Net na CPU. Nie sprawdzają instalacji cuML ani jakości modeli na rzeczywistych zdjęciach.

## Przygotowanie danych

Przetwarzanie danych wykonano przy użyciu Pythona 3.13 i NumPy 2.2.4. Dla zachowania tych wersji użyj osobnego środowiska:

```bash
python3.13 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt
python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
python -m preprocessing.prepare catalog
python -m preprocessing.prepare composites
python -m preprocessing.package
```

Pobieranie HLS wymaga własnego konta [NASA Earthdata](https://urs.earthdata.nasa.gov/). Biblioteka zapisuje dane logowania w pliku użytkownika `.netrc` lub `_netrc`, poza repozytorium, zgodnie z [instrukcją uwierzytelniania earthaccess](https://earthaccess.readthedocs.io/en/latest/user/howto/authenticate/). MapBiomas jest odczytywany z publicznego adresu podanego w konfiguracji.

Katalog wyszukiwania jest ograniczany do identyfikatorów scen zapisanych w badaniu. Kompozyty powstają dla ustalonego podziału obszarów. Do przetestowania samego pobierania i przetwarzania jednego obrazu można użyć `python -m preprocessing.prepare composites --area PA_T01 --year 2015`. Utworzenie całego pakietu wymaga wszystkich 28 obrazów.

Kod wyboru obszarów jest dostępny jako etapy `baseline`, `qa` i `select` modułu `preprocessing.prepare`. Nie są potrzebne do użycia dołączonego podziału. Etap `select` sprawdza zgodność z zapisanymi obszarami i nie nadpisuje innego podziału.

Reguły przetwarzania: maskowanie chmur, cieni, sąsiedztwa chmur, śniegu/lodu, wysokiego aerozolu i niepoprawnych wartości pasm. Dla danego dnia zachowywany jest pierwszy kompletny wektor sześciu pasm według kolejności identyfikatorów scen. Następnie obliczana jest mediana każdego pasma i wskaźniki NDVI, NDMI oraz NBR. Wymagane są co najmniej trzy poprawne dni. Kody MapBiomas 3, 5, 6 i 49 oznaczają las, pozostałe rozpoznane klasy oznaczają nie-las (w tym plantacje, kod 9). Braki i nierozpoznane kody są wykluczane.

Wyniki przygotowania trafiają do `data/prepared/`, a pakiet wejściowy do `data/bundles/landsat8_jun_sep_v2_r2/`. Testowe pliki NPZ nie zawierają etykiet. Mapy referencyjne testu pozostają w `data/prepared/test/` i są odczytywane dopiero przez ocenę. Pobieranie pełnego zbioru jest znacznie dłuższe od testu instalacji i wymaga miejsca na rastry oraz pamięć podręczną.

## Trening i ocena

Uczenie wykonano na Pythonie 3.11, z wersjami bibliotek zapisanymi w `requirements.txt`. RF korzysta z CPU, a SVM z cuML i U-Net z PyTorch na GPU NVIDIA. Dla tych dwóch metod wymagane jest środowisko Linux lub odpowiednio skonfigurowany WSL2. Wymagania instalacyjne cuML opisuje [dokumentacja RAPIDS](https://docs.rapids.ai/install/).

W środowisku treningowym z działającym sterownikiem NVIDIA:

```bash
source .venv-train/bin/activate
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-svm.txt
python -m training.run --family rf_ndvi --data data/bundles/landsat8_jun_sep_v2_r2 --output outputs/main/rf_ndvi
python -m training.run --family unet --data data/bundles/landsat8_jun_sep_v2_r2 --output outputs/main/unet
python -m training.cuda -m training.run --family svm --data data/bundles/landsat8_jun_sep_v2_r2 --output outputs/main/svm
python -m evaluation.run --data data/bundles/landsat8_jun_sep_v2_r2 --prepared data/prepared --runs outputs/main/rf_ndvi outputs/main/unet outputs/main/svm --output outputs/evaluation --maps
```

Moduł `training.cuda` udostępnia biblioteki CUDA z pakietów Pythona w układzie wymaganym przez CuPy/Numba. Sam RF można uruchomić z `requirements.txt`, bez cuML i GPU. Rodziny metod można uruchamiać oddzielnie, na różnych komputerach, a następnie przekazać ich katalogi wynikowe do jednej oceny.

Konfiguracje i kolejność kandydatów odpowiadają eksperymentowi. Dobór parametrów odbywa się na podstawie średniego F1 ubytku na dwóch obszarach walidacyjnych. RF i U-Net mają trzy końcowe inicjalizacje (42, 123, 2026), SVM jedną. Wszystkie modele danej rodziny zostają ustalone przed odczytem obrazów testowych. Łącznie wykonywanych jest 19 dopasowań modeli oraz dobór progu NDVI.

Ocena jakości obejmuje okresy 2015-2020 i 2020-2025. Powierzchnie są dodatkowo obliczane na wspólnym zakresie trzech lat, także bezpośrednio dla 2015-2025. Porównanie dekady nie stanowi trzeciego okresu głównej oceny jakości. Kodowanie zmian: 0 - stabilny nie-las, 1 - przyrost, 2 - ubytek, 3 - stabilny las, 255 - brak danych. Jeden piksel odpowiada 0,09 ha. Miary zbiorcze nadają równe wagi obszarom, okresom i inicjalizacjom. Macierze pomyłek zapisują liczby pikseli.

Każde nowe uruchomienie wymaga pustego katalogu wynikowego. Główny trening można wznowić do nowego katalogu przez `--resume-dir <poprzedni-katalog>`. Wagi i pliki `.joblib` należy odczytywać wyłącznie z własnych, zaufanych obliczeń.

## Analiza dodatkowych etykiet i epok

```bash
python -m experiments.prepare --data data/bundles/landsat8_jun_sep_v2_r2 --prepared data/prepared --output data/controls
python -m training.cuda -m experiments.run --data data/bundles/landsat8_jun_sep_v2_r2 --samples data/controls/nested_samples.npz --output outputs/controls
python -m experiments.analyze --data data/bundles/landsat8_jun_sep_v2_r2 --runs outputs/controls --output outputs/controls_evaluation
```

Większe próby zachowują wcześniejsze punkty i używają tych samych 12 obrazów. Normalizacja pozostaje wyznaczona na pierwotnych 60 tys. etykiet. Plan obejmuje 27 dopasowań: 60/120/240 tys. etykiet przy ustalonych parametrach oraz U-Net do 100 epok ze stałym lub zmniejszanym współczynnikiem uczenia. Opcja `--method rf`, `--method svm` lub `--method unet` pozwala uruchomić jedną metodę w osobnym katalogu. Analiza ta używa wyłącznie walidacji i nie zastępuje wyników testu głównego.

## Wyniki do porównania

`results/main_comparison.csv` zawiera ocenę testową, a `results/controls_comparison.csv` ocenę dodatkowych eksperymentów na walidacji. Wartości miar zapisano w skali 0-1. Pliki pochodzą z eksperymentów opisanych w pracy, a nie z małych testów repozytorium. Średnie F1 ubytku w głównym teście wynosi: NDVI 44,9%, RF 52,4%, SVM 52,3%, U-Net 60,9%.

Kod wyodrębniono z implementacji tych eksperymentów. Odtwarzalność liczb zależy również od dostępności tych samych wersji danych i środowiska obliczeń. Obliczenia GPU mogą wykazywać niewielką zmienność między uruchomieniami.
