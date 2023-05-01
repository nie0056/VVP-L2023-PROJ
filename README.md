# Repozitář do předmětu VVP, sloužící k odevzdání projektu

## Popis projektu

**Zadání:** balíček *gradsolv*

Cílem projektu byla implementace balíčku, 
poskytujícího možnost řešit soustavu lineárních rovnic se symetrickou pozitivně definitní maticí pomocí dvou gradientních iteračních metod, 
**metody největšího spádu** a **metody sdružených gradientů**.

Dalším z úkolů byla implementace tří předpodmíňovačů, a to **diagonálního** a dvou variant **Gauss-Seidel** předpodmíňovače.

Součástí implementace daných řešičů je i základní kontrola správnosti vstupů:
- zda je matice soustavy opravdu maticí,
- zda je matice na vstupu čtvercová,
- zda je matice na vstupu symetrická,
- zda má matice všechny prvky na hlavní diagonále kladné,
- zda je vektor pravé strany opravdu vektorem,
- zda rozměr čtvercové matice soustavy je shodný s rozměrem vektoru pravé strany.

Oba řešiče také zaznamenávají, jako chybový stav, situaci, kdy byl proveden maximální počet iterací, avšak přibližné řešení nedosáhlo požadované přesnosti.

K tomu, aby balíček *gradsolv* fungoval správně, je třeba mít nainstalovaný balíček *numpy*.

## Použití

Použití je ukázáno v souboru **gradsolv_demo.ipynb**, jehož obsahem je demonstrace funcionality daného balíčku.