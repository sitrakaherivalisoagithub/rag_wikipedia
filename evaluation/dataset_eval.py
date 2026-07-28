EVAL_DATASET = [
    {
        "id": "Q01",
        "category": "Fait simple",
        "question": "Quelle est la monnaie officielle de Madagascar ?",
        "ground_truth_answer": "L'Ariary (MGA).",
        "ground_truth_source": "Introduction / Economy"
    },
    {
        "id": "Q02",
        "category": "Chiffre précis",
        "question": "Combien de régions composent l'organisation administrative de Madagascar ?",
        "ground_truth_answer": "23 régions (ou 24 depuis la récente création de la région Vatovavy).",
        "ground_truth_source": "Administrative divisions"
    },
    {
        "id": "Q03",
        "category": "Lecture de tableau",
        "question": "Quelle est la capitale et la population de la région Analamanga ?",
        "ground_truth_answer": "La capitale est Antananarivo et la population est d'environ 3 624 000 habitants.",
        "ground_truth_source": "Administrative divisions (Regions table)"
    },
    {
        "id": "Q04",
        "category": "Lecture de tableau",
        "question": "Quel groupe ethnique représente environ 26% de la population malgache ?",
        "ground_truth_answer": "Le groupe Merina.",
        "ground_truth_source": "Demographics (Ethnic groups table)"
    },
    {
        "id": "Q05",
        "category": "Multi-passages",
        "question": "Comment s'est faite la transition vers la colonisation française à la fin du XIXe siècle ?",
        "ground_truth_answer": "Après les guerres franco-merina, la France a déclaré l'île colonie en 1896, a aboli la monarchie merina et a exilé la reine Ranavalona III en 1897.",
        "ground_truth_source": "History (French Madagascar)"
    },
    {
        "id": "Q06",
        "category": "Multi-passages",
        "question": "Quelles sont les conséquences environnementales de la pratique du tavy (agriculture sur brûlis) ?",
        "ground_truth_answer": "La déforestation accélérée, l'érosion des sols et la perte de biodiversité/habitats pour les espèces endémiques.",
        "ground_truth_source": "Ecology / Environment"
    },
    {
        "id": "Q07",
        "category": "Ambiguïté temporelle",
        "question": "Qui a été le premier président de la République malgache après l'indépendance de 1960 ?",
        "ground_truth_answer": "Philibert Tsiranana.",
        "ground_truth_source": "History (Independence)"
    },
    {
        "id": "Q08",
        "category": "Hors périmètre (Piège)",
        "question": "Quel est le nom du volcan en éruption active situé à Antananarivo ?",
        "ground_truth_answer": "Information absente de la page / Il n'y a pas de volcan en éruption active à Antananarivo. L'agent doit indiquer qu'il ne trouve pas cette donnée.",
        "ground_truth_source": "N/A (Hors périmètre)"
    },
    {
        "id": "Q09",
        "category": "Partiellement couverte",
        "question": "Quel est le climat à Madagascar ?",
        "ground_truth_answer": "Le climat varie selon les régions : tropical le long de la côte est, aride/semi-aride au sud, et tempéré/frais sur les hautes terres centrales.",
        "ground_truth_source": "Geography (Climate)"
    },
    {
        "id": "Q10",
        "category": "Hors périmètre (Piège)",
        "question": "Combien de médailles d'or olympiques Madagascar a-t-elle remportées en natation ?",
        "ground_truth_answer": "L'information est absente de la page Wikipédia. L'agent doit déclarer ne pas avoir l'information.",
        "ground_truth_source": "N/A (Hors périmètre)"
    }
]