"""
generate_bias_fakes.py — Generate TTS fakes for bias audit across 6 languages.

Uses FREE TTS engines only:
  - edge-tts (Microsoft, async): Hausa, French, Arabic, English, Pidgin (via en-NG)
  - gTTS (Google): Yoruba, fallback for edge-tts failures

Run LOCALLY (not on Kaggle — needs network access):
    pip install edge-tts gtts pydub
    python generate_bias_fakes.py --output-dir ./bias_audit_fakes
    python generate_bias_fakes.py --output-dir ./bias_audit_fakes --language yoruba  # single lang
    python generate_bias_fakes.py --list-voices  # check available voices

Output structure:
    bias_audit_fakes/
      english/fake/   (50 clips)
      yoruba/fake/    (50 clips)
      hausa/fake/     (50 clips)
      pidgin/fake/    (50 clips)
      french/fake/    (50 clips)
      arabic/fake/    (50 clips)
      manifest.json   (all entries with path/label/language/tts_engine/voice/gender)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ===================================================================
# TEXT SAMPLES PER LANGUAGE
# ===================================================================
# ~60 sentences per language to generate 50 clips (buffer for failures)

TEXTS = {
    "english": [
        "The economic outlook for Nigeria continues to show signs of growth.",
        "Farmers in the northern region are preparing for the planting season.",
        "The government has announced new infrastructure projects across the country.",
        "Technology companies are expanding their operations in Lagos.",
        "Education reform remains a top priority for the administration.",
        "The central bank has adjusted interest rates to control inflation.",
        "Healthcare workers are being trained in rural communities.",
        "New trade agreements have been signed with neighboring countries.",
        "The entertainment industry continues to gain international recognition.",
        "Water supply projects are being completed in several states.",
        "The transportation network is being modernized with new rail lines.",
        "Young entrepreneurs are creating innovative solutions for local challenges.",
        "The agricultural sector contributes significantly to the national economy.",
        "Security forces are working to maintain peace in affected regions.",
        "Cultural festivals celebrate the diversity of Nigerian traditions.",
        "International investors are showing increased interest in the market.",
        "The telecommunications sector has experienced rapid growth.",
        "Environmental conservation efforts are gaining momentum.",
        "Sports development programs are being expanded nationwide.",
        "The judiciary is implementing reforms to improve efficiency.",
        "Housing development projects aim to address the urban deficit.",
        "Renewable energy initiatives are being launched in several states.",
        "The manufacturing sector is growing with new industrial zones.",
        "Digital literacy programs are reaching more communities.",
        "Public health campaigns are improving awareness of preventable diseases.",
        "The aviation industry is recovering with new route expansions.",
        "Financial inclusion efforts are reaching previously unbanked populations.",
        "Research institutions are producing groundbreaking scientific work.",
        "The creative economy is becoming a major contributor to GDP.",
        "Road construction projects are improving connectivity between cities.",
        "Maritime trade continues to expand through major port upgrades.",
        "Social protection programs are being strengthened for vulnerable groups.",
        "The mining sector is being developed with proper regulatory frameworks.",
        "Urban planning initiatives are addressing rapid population growth.",
        "Diplomatic relations are being strengthened across the African continent.",
        "The insurance industry is growing with new product offerings.",
        "Vocational training centers are producing skilled workers.",
        "Climate adaptation strategies are being developed for coastal areas.",
        "The real estate market is showing signs of recovery.",
        "Youth empowerment programs are creating new opportunities.",
        "The power sector is being reformed to improve electricity supply.",
        "Agricultural research is developing drought resistant crop varieties.",
        "Tourism promotion campaigns are highlighting natural attractions.",
        "The pension system is being reformed to ensure sustainability.",
        "Cross border trade is facilitated by improved customs procedures.",
        "Innovation hubs are supporting technology startups.",
        "The pharmaceutical industry is expanding local production capacity.",
        "Community development programs are transforming rural areas.",
        "The stock exchange has seen increased trading activity.",
        "Media organizations are adapting to the digital transformation.",
        "Flood management systems are being installed in vulnerable areas.",
        "The fashion industry is gaining global recognition.",
        "Cooperative societies are empowering small scale farmers.",
        "Public transportation systems are being upgraded in major cities.",
        "The hospitality industry is attracting more international visitors.",
        "Anti corruption measures are being strengthened across institutions.",
        "Science and technology parks are driving innovation.",
        "The livestock industry is being modernized with better practices.",
        "Coastal erosion prevention measures are being implemented.",
        "The banking sector remains resilient despite global challenges.",
    ],
    "yoruba": [
        "Ọjọ́ oni dára fún gbogbo ènìyàn ní ilẹ̀ Nàìjíríà.",
        "Àwọn àgbẹ̀ ń múra sílẹ̀ fún àkókò ìrúgbìn tuntun.",
        "Ìjọba ti kéde àwọn iṣẹ́ àgbékalẹ̀ tuntun ní orílẹ̀ èdè.",
        "Ẹ̀kọ́ jẹ́ pàtàkì fún ìdàgbàsókè orílẹ̀ èdè wa.",
        "Ilé ìwòsàn tuntun ti ṣí sílẹ̀ ní àárín ìlú.",
        "Àwọn ọmọ ilé ẹ̀kọ́ ń kẹ́kọ̀ọ́ dáadáa ní ilé ẹ̀kọ́.",
        "Oúnjẹ tó dára jẹ́ pàtàkì fún ìlera ara.",
        "Omi mímọ́ jẹ́ ohun pàtàkì fún gbogbo àwùjọ.",
        "Àwọn oṣiṣẹ́ ń ṣiṣẹ́ láàárín fún ìdàgbàsókè ọrọ̀ ajé.",
        "Ìṣèlú jẹ́ ohun pàtàkì nínú ìgbésí ayé wa.",
        "Ẹ̀sìn kọ̀ọ̀kan ní àwọn ẹ̀kọ́ rẹ̀ tó dára.",
        "Ọjà tuntun ti ṣí sílẹ̀ ní àdúgbò wa.",
        "Àwọn ọ̀dọ́ ń ṣiṣẹ́ takuntakun fún ọjọ́ ọ̀la wọn.",
        "Ìrìn àjò sí ibì kan pàtàkì jẹ́ ohun tí ó dára.",
        "Ọkọ̀ ojú irin tuntun ti bẹ̀rẹ̀ ní ìlú wa.",
        "Àṣà àti ìṣe wa jẹ́ ohun ìyìn fún gbogbo wa.",
        "Iṣẹ́ ọnà jẹ́ ọ̀nà kan tí a fi ń sọ àṣà wa di mímọ́.",
        "Ẹ̀rọ tuntun ń ran àwọn àgbẹ̀ lọ́wọ́ nínú iṣẹ́ wọn.",
        "Ìdájọ́ tó tọ́ jẹ́ pàtàkì fún àlàáfíà ìlú.",
        "Orin àti eré jẹ́ apá pàtàkì nínú àṣà wa.",
        "Ẹ̀kọ́ gíga jẹ́ ohun pàtàkì fún àwọn ọ̀dọ́.",
        "Àwọn obìnrin ń kópa nínú ìdàgbàsókè orílẹ̀ èdè.",
        "Ìmọ̀ ẹ̀rọ tuntun ń yí ayé wa padà.",
        "Owó orílẹ̀ èdè wa ń lágbára sí i.",
        "Àwọn ọmọdé ni ọjọ́ ọ̀la orílẹ̀ èdè.",
        "Iṣẹ́ àgbẹ̀ jẹ́ iṣẹ́ pàtàkì ní ilẹ̀ Nàìjíríà.",
        "Ìlera àwọn ènìyàn jẹ́ ohun tí ó ṣe pàtàkì jùlọ.",
        "Ìwà ọmọlúwàbí jẹ́ ohun tí ó dára fún gbogbo wa.",
        "Àwọn aṣáájú ìlú ń ṣiṣẹ́ fún àǹfààní gbogbo ènìyàn.",
        "Ilẹ̀ Nàìjíríà jẹ́ orílẹ̀ èdè tó tóbi jùlọ ní Áfíríkà.",
        "Omi òjò jẹ́ àǹfààní fún àwọn àgbẹ̀.",
        "Ìgbàgbọ́ jẹ́ ohun pàtàkì nínú ìgbésí ayé.",
        "Àwọn ọkùnrin àti obìnrin ní ẹ̀tọ́ kan náà.",
        "Ìmọ̀ sáyẹ́nsì ń mú ìlọsíwájú bá ayé.",
        "Ẹ̀dá ènìyàn gbọ́dọ̀ bọ̀wọ̀ fún ara wọn.",
        "Iṣẹ́ àkànṣe ń mú ìdàgbàsókè bá ọrọ̀ ajé.",
        "Àwọn ìlú ńlá ń dàgbà ní gbogbo ọjọ́.",
        "Ìfọwọ́sowọ́pọ̀ jẹ́ ohun pàtàkì fún ìlọsíwájú.",
        "Ẹ̀kọ́ ọ̀fẹ́ jẹ́ ẹ̀tọ́ gbogbo ọmọdé.",
        "Iṣẹ́ ìjọba jẹ́ iṣẹ́ fún àǹfààní gbogbo ènìyàn.",
        "Àwọn ọjọ́ ìsinmi jẹ́ àkókò ìsinmi fún gbogbo wa.",
        "Ìwà rere jẹ́ ohun tí ó dára jùlọ nínú ìgbésí ayé.",
        "Orílẹ̀ èdè wa ní ohun àmúṣọrọ̀ púpọ̀.",
        "Ẹ̀kọ́ nínú èdè Yorùbá jẹ́ ohun pàtàkì.",
        "Àwọn ọmọ wa gbọ́dọ̀ mọ̀ ìtàn orílẹ̀ èdè wọn.",
        "Iṣẹ́ ọwọ́ jẹ́ ọ̀nà kan sí ìdàgbàsókè ọrọ̀ ajé.",
        "Àjọ àgbáyé ń ṣiṣẹ́ fún àlàáfíà àgbáyé.",
        "Ìdúróṣinṣin jẹ́ ohun pàtàkì fún àṣeyọrí.",
        "Àwọn ẹranko igbó jẹ́ ohun tí a gbọ́dọ̀ dáàbò bò.",
        "Ilé tí ó dára jẹ́ ohun pàtàkì fún ẹbí.",
        "Àwọn ènìyàn ń gbàdúrà fún àlàáfíà orílẹ̀ èdè.",
        "Ìfẹ́ jẹ́ ohun pàtàkì jùlọ nínú ìgbésí ayé ènìyàn.",
        "Ọjọ́ ọ̀la wa yóò dára ju oni lọ.",
        "Iṣẹ́ tí a bá fẹ́ ṣe ní a gbọ́dọ̀ ṣe dáadáa.",
        "Àwọn àgbàlagbà jẹ́ ènìyàn pàtàkì nínú àwùjọ.",
        "Ọmọ tí a kọ́ dáadáa máa ń dára.",
        "Ìrètí jẹ́ ohun tí ó mú kí a máa tẹ̀síwájú.",
        "Ẹ̀kọ́ kò ní ìparí nínú ìgbésí ayé ènìyàn.",
        "Ìṣọ̀kan jẹ́ agbára fún orílẹ̀ èdè.",
        "Àwọn ohun ọ̀ṣọ́ wa jẹ́ ohun ìyìn fún àṣà wa.",
    ],
    "hausa": [
        "Nijeriya kasa ce mai girma a nahiyar Afirka.",
        "Manoma suna shirya don lokacin shuka na sabon damina.",
        "Gwamnati ta sanar da sabbin ayyukan gine-gine a kasar.",
        "Ilimi shi ne mabudin ci gaba a rayuwa.",
        "Asibitoci suna samun ingantattun kayan aiki.",
        "Tattalin arzikin kasar yana ci gaba da bunkasa.",
        "Matasa suna kokarin neman ilimi don inganta rayuwarsu.",
        "Aikin gona yana da muhimmanci ga tattalin arzikin kasa.",
        "Lafiyar jama'a ita ce abu mafi muhimmanci.",
        "Zaman lafiya yana da muhimmanci ga ci gaban kasa.",
        "Mata suna taka rawa sosai wajen ci gaban al'umma.",
        "Fasahar zamani tana canza rayuwar mutane.",
        "Kasuwanci yana bunkasa a manyan birane.",
        "Ruwan sha mai tsabta yana da muhimmanci ga lafiya.",
        "Yara su ne gaba da fatan kasa.",
        "Addini yana koyar da kyawawan halaye.",
        "Hakkin dan Adam ya kamata a mutunta shi.",
        "Shugabanni suna aiki don amfanin jama'a.",
        "Wasanni suna hada kan matasa.",
        "Makarantun gwamnati suna bukata ingantawa.",
        "Kiwo yana daya daga cikin manyan sana'o'i.",
        "Hanyoyin mota suna bukatar gyara.",
        "Kamfanonin sadarwa suna bunkasa cikin sauri.",
        "Masana'antu suna samar da ayyukan yi ga matasa.",
        "Albarkatun kasa sun hada da man fetur da iskar gas.",
        "Ruwan sama yana da amfani ga aikin gona.",
        "Shirin ci gaban karkara yana taimaka wa manoma.",
        "Jami'o'i suna horar da dalibai a fannoni daban-daban.",
        "Bangaren kiwon lafiya yana bukatar karin ma'aikata.",
        "Tsaron kasa yana da muhimmanci ga zaman lafiya.",
        "Gidajen rediyo suna yada labarai ga jama'a.",
        "Hukumar zabe tana shirya zaben gaba.",
        "Kasashen waje suna zuba jari a Nijeriya.",
        "Noma na zamani yana amfani da injuna.",
        "Tsarin shari'a yana bukatar gyara.",
        "Filin jirgin sama na bukatar fadada.",
        "Bankunan kasa suna ba da lamuni ga matasa.",
        "Ayyukan jin kai suna taimaka wa marasa galihu.",
        "Cibiyoyin bincike suna gudanar da bincike mai muhimmanci.",
        "Masana kimiyya suna gano sabbin hanyoyin magance cututtuka.",
        "Kudi na kasa yana bukatar karfafawa.",
        "Yawon bude ido yana kawo kudade ga kasa.",
        "Al'adun gargajiya suna da muhimmanci ga al'umma.",
        "Kamfanonin mota suna fadada kasuwancinsu.",
        "Motocin lantarki suna zuwa kasuwa.",
        "Makamashi mai sabuntawa yana da muhimmanci.",
        "Shirin rage talauci yana taimaka wa jama'a.",
        "Sauyin yanayi yana shafar aikin gona.",
        "Fasahar sadarwa tana hada kan mutane.",
        "Ci gaban kasa yana bukatar hadin kan kowa.",
        "Matasan Nijeriya suna da kwarewa sosai.",
        "Harshen Hausa yana daya daga cikin manyan harsunan Afirka.",
        "Kananan hukumomi suna da alhakin ci gaban yankunansu.",
        "Binciken kimiyya yana samar da sabbin ilimomi.",
        "Kafofin yada labarai suna da muhimmanci a dimokuradiyya.",
        "Masana'antar fina-finai ta Hausa tana bunkasa.",
        "Shirin tallafin matasa yana samar da ayyukan yi.",
        "Kamfanonin fasaha suna zuwa Nijeriya.",
        "Rayuwar karkara ta bambanta da ta birni.",
        "Nijeriya tana da yawan jama'a mafi girma a Afirka.",
    ],
    "pidgin": [
        "How work dey go for you today my broda?",
        "Dis country go better if we all work together.",
        "Plenty people dey find work for Lagos every day.",
        "Na education be the key to success for life.",
        "The government don promise say dem go fix the roads.",
        "Water no dey come for our area since last week.",
        "Market women dey sell their goods for the roadside.",
        "The price of food don increase well well this year.",
        "Young people dey hustle hard to make money.",
        "Na God dey protect us every day for this country.",
        "The new hospital wey dem build don start to dey work.",
        "Farmer people dey prepare for the new planting season.",
        "Everybody need good health to enjoy life well.",
        "The children dem need better school to attend.",
        "Technology don change the way people dey do business.",
        "Na together we go build this country make e better.",
        "Some people dey leave the village come city find work.",
        "Rain don fall well well this year for the south.",
        "The price of petrol don go up again for filling station.",
        "Mama dey cook food for the house every morning.",
        "The electricity no dey come regularly for our area.",
        "Small small business people dey contribute to the economy.",
        "Na patience and hard work go bring success.",
        "The football match wey we watch yesterday sweet well well.",
        "Transportation problem dey worry people for the big cities.",
        "Bank people say dem go give loan to small business.",
        "People dey celebrate festival with plenty food and music.",
        "The new road wey dem build don make travel easy.",
        "Doctors and nurses dey work hard for the hospital.",
        "Children wey go school go get better future.",
        "The weather don dey hot well well this dry season.",
        "Fishermen dey catch fish for the river every morning.",
        "Tailors dey sew plenty clothes for the market.",
        "Musicians dey sing songs wey touch people heart.",
        "The community dey come together help each other.",
        "Na respect and love dey keep family together.",
        "Mobile phone don make communication easy for everybody.",
        "Teachers dey work hard but dem no dey get enough pay.",
        "The bridge wey dem build don connect the two villages.",
        "Young girls need education same way like boys.",
        "Construction workers dey build new houses everywhere.",
        "People dey travel from one state to another for business.",
        "The market dey open early morning close for evening.",
        "Palm oil na important thing for Nigerian cooking.",
        "Women dey contribute well well to the family income.",
        "Mechanic people dey fix motor for the roadside.",
        "Prayer dey help people pass through difficult times.",
        "The river dey flow well well during rainy season.",
        "Okada riders dey carry people go where dem want.",
        "This year harvest go better pass last year own.",
        "People need clean water for drink and for cook.",
        "The town chief don call meeting for the community.",
        "Carpenters dey make fine furniture for people house.",
        "Street lights dey help make the road safe for night.",
        "Wedding ceremony don hold for the church last Saturday.",
        "Pepper and tomato na important market goods.",
        "Night market dey give people chance to buy things cheap.",
        "Village people dey welcome visitors with open arms.",
        "The new policy go help small businesses grow well.",
        "Everybody dey hope say things go change for better.",
    ],
    "french": [
        "Le développement économique du Nigeria continue de progresser.",
        "Les agriculteurs se préparent pour la nouvelle saison des pluies.",
        "Le gouvernement a annoncé de nouveaux projets d'infrastructure.",
        "L'éducation est la clé du progrès pour chaque nation.",
        "Les hôpitaux reçoivent de nouveaux équipements médicaux.",
        "La technologie transforme la manière dont les gens travaillent.",
        "Les jeunes entrepreneurs créent des solutions innovantes.",
        "La santé publique est une priorité pour le pays.",
        "Les femmes jouent un rôle important dans le développement.",
        "Le commerce international favorise la croissance économique.",
        "Les universités forment les futurs dirigeants du pays.",
        "La culture africaine est riche et diversifiée.",
        "Les énergies renouvelables sont essentielles pour l'avenir.",
        "Le sport rassemble les communautés et renforce les liens.",
        "La recherche scientifique produit des résultats remarquables.",
        "Les droits de l'homme doivent être respectés partout.",
        "La paix est nécessaire pour le progrès de chaque nation.",
        "Les infrastructures de transport sont en cours de modernisation.",
        "Le secteur bancaire est en pleine transformation numérique.",
        "Les communautés rurales bénéficient de programmes de développement.",
        "La conservation de l'environnement est cruciale pour les générations futures.",
        "Les festivals culturels célèbrent la diversité des traditions.",
        "L'agriculture reste un pilier de l'économie nationale.",
        "Les télécommunications ont révolutionné la vie quotidienne.",
        "La justice sociale est un objectif fondamental de la société.",
        "Les artistes africains gagnent une reconnaissance internationale.",
        "Le tourisme contribue significativement au produit intérieur brut.",
        "Les programmes de formation professionnelle préparent les jeunes.",
        "La coopération régionale favorise le développement mutuel.",
        "Les marchés financiers montrent des signes de stabilité.",
        "La diplomatie joue un rôle essentiel dans les relations internationales.",
        "Les innovations technologiques améliorent la productivité agricole.",
        "Le système éducatif nécessite des réformes importantes.",
        "Les organisations non gouvernementales soutiennent les communautés vulnérables.",
        "La sécurité alimentaire est un enjeu majeur pour le continent.",
        "Les investissements étrangers contribuent à la création d'emplois.",
        "Le patrimoine culturel doit être préservé et valorisé.",
        "Les médias jouent un rôle crucial dans la démocratie.",
        "La transition énergétique est un défi mondial.",
        "Les petites et moyennes entreprises sont le moteur de l'économie.",
        "Le changement climatique affecte les populations les plus vulnérables.",
        "Les droits des femmes progressent grâce à l'action collective.",
        "La recherche médicale fait des progrès considérables chaque année.",
        "Les infrastructures numériques sont essentielles au développement.",
        "Le dialogue interculturel favorise la compréhension mutuelle.",
        "Les programmes sociaux aident les familles dans le besoin.",
        "La bonne gouvernance est essentielle pour la confiance publique.",
        "Les avancées scientifiques ouvrent de nouvelles perspectives.",
        "Le secteur privé contribue à la diversification de l'économie.",
        "L'accès à l'eau potable est un droit fondamental.",
        "Les projets d'urbanisation visent à améliorer la qualité de vie.",
        "La formation continue est importante pour rester compétitif.",
        "Les industries créatives génèrent des emplois et de la richesse.",
        "Le système de santé publique nécessite des investissements importants.",
        "Les relations diplomatiques renforcent la coopération internationale.",
        "La protection de la biodiversité est une responsabilité partagée.",
        "Les technologies de l'information transforment tous les secteurs.",
        "L'entrepreneuriat social apporte des solutions aux problèmes communautaires.",
        "Le développement durable est l'objectif de toutes les nations.",
        "L'unité nationale est la force de chaque pays.",
    ],
    "arabic": [
        "التنمية الاقتصادية في نيجيريا تشهد تقدماً ملحوظاً.",
        "المزارعون يستعدون لموسم الزراعة الجديد.",
        "الحكومة أعلنت عن مشاريع بنية تحتية جديدة.",
        "التعليم هو أساس التقدم والتنمية في أي مجتمع.",
        "المستشفيات تحصل على معدات طبية حديثة.",
        "التكنولوجيا تغير طريقة حياة الناس بشكل كبير.",
        "الشباب يبتكرون حلولاً جديدة للتحديات المحلية.",
        "الصحة العامة هي أولوية قصوى للحكومة.",
        "المرأة تلعب دوراً محورياً في تنمية المجتمع.",
        "التجارة الدولية تعزز النمو الاقتصادي للبلاد.",
        "الجامعات تخرج قادة المستقبل في مختلف المجالات.",
        "الثقافة الأفريقية غنية ومتنوعة بتراثها العريق.",
        "الطاقة المتجددة ضرورية لمستقبل مستدام.",
        "الرياضة توحد المجتمعات وتعزز الروابط بين الناس.",
        "البحث العلمي ينتج نتائج مهمة ومبتكرة.",
        "حقوق الإنسان يجب أن تُحترم في كل مكان.",
        "السلام ضروري لتحقيق التقدم والازدهار.",
        "شبكات النقل يتم تحديثها بمشاريع جديدة.",
        "القطاع المصرفي يمر بتحول رقمي شامل.",
        "المجتمعات الريفية تستفيد من برامج التنمية المحلية.",
        "الحفاظ على البيئة أمر بالغ الأهمية للأجيال القادمة.",
        "المهرجانات الثقافية تحتفي بتنوع التقاليد والعادات.",
        "الزراعة تبقى ركيزة أساسية في الاقتصاد الوطني.",
        "الاتصالات غيرت الحياة اليومية بشكل جذري.",
        "العدالة الاجتماعية هدف أساسي يسعى إليه المجتمع.",
        "الفنانون الأفارقة يحظون باعتراف دولي متزايد.",
        "السياحة تساهم بشكل كبير في الناتج المحلي.",
        "برامج التدريب المهني تؤهل الشباب لسوق العمل.",
        "التعاون الإقليمي يعزز التنمية المشتركة بين الدول.",
        "الأسواق المالية تظهر علامات استقرار ونمو.",
        "الدبلوماسية تلعب دوراً أساسياً في العلاقات الدولية.",
        "الابتكارات التقنية تحسن الإنتاجية في مختلف القطاعات.",
        "النظام التعليمي يحتاج إلى إصلاحات جوهرية.",
        "المنظمات غير الحكومية تدعم المجتمعات الأكثر ضعفاً.",
        "الأمن الغذائي قضية محورية تواجه القارة الأفريقية.",
        "الاستثمارات الأجنبية تساهم في خلق فرص عمل جديدة.",
        "التراث الثقافي يجب الحفاظ عليه وتعزيزه.",
        "وسائل الإعلام تؤدي دوراً حاسماً في الديمقراطية.",
        "التحول في مجال الطاقة يمثل تحدياً عالمياً.",
        "المؤسسات الصغيرة والمتوسطة هي محرك الاقتصاد.",
        "التغير المناخي يؤثر على الفئات الأكثر ضعفاً.",
        "حقوق المرأة تتقدم بفضل الجهود الجماعية المستمرة.",
        "البحث الطبي يحقق تقدماً كبيراً كل عام.",
        "البنية التحتية الرقمية أساسية للتنمية الحديثة.",
        "الحوار بين الثقافات يعزز التفاهم والتعايش المشترك.",
        "البرامج الاجتماعية تساعد الأسر المحتاجة والفقيرة.",
        "الحوكمة الرشيدة ضرورية لبناء ثقة المواطنين.",
        "التقدم العلمي يفتح آفاقاً جديدة للبشرية.",
        "القطاع الخاص يساهم في تنويع مصادر الاقتصاد.",
        "الحصول على مياه نظيفة حق أساسي لكل إنسان.",
        "مشاريع التحضر تهدف إلى تحسين جودة الحياة.",
        "التعلم المستمر مهم للبقاء في سوق العمل.",
        "الصناعات الإبداعية تولد فرص عمل وثروة.",
        "نظام الرعاية الصحية يحتاج استثمارات كبيرة.",
        "العلاقات الدبلوماسية تقوي التعاون بين الشعوب.",
        "حماية التنوع البيولوجي مسؤولية مشتركة بين الجميع.",
        "تقنيات المعلومات تحول جميع القطاعات الاقتصادية.",
        "ريادة الأعمال الاجتماعية تقدم حلولاً للمشاكل المجتمعية.",
        "التنمية المستدامة هي هدف جميع الأمم والشعوب.",
        "الوحدة الوطنية هي قوة كل دولة ومجتمع.",
    ],
}

# ===================================================================
# TTS VOICE CONFIGS
# ===================================================================
EDGE_TTS_VOICES = {
    "english": [
        {"name": "en-NG-AbeoNeural", "gender": "male"},
        {"name": "en-NG-EzinneNeural", "gender": "female"},
    ],
    "hausa": [
        {"name": "ha-NG-JibrilNeural", "gender": "male"},
        {"name": "ha-NG-HadizaNeural", "gender": "female"},
    ],
    "pidgin": [
        # No dedicated Pidgin voices — use Nigerian English
        {"name": "en-NG-AbeoNeural", "gender": "male"},
        {"name": "en-NG-EzinneNeural", "gender": "female"},
    ],
    "french": [
        {"name": "fr-FR-HenriNeural", "gender": "male"},
        {"name": "fr-FR-DeniseNeural", "gender": "female"},
    ],
    "arabic": [
        {"name": "ar-SA-HamedNeural", "gender": "male"},
        {"name": "ar-SA-ZariyahNeural", "gender": "female"},
    ],
}

GTTS_LANGS = {
    "yoruba": "yo",
    "english": "en",
    "hausa": "ha",
    "french": "fr",
    "arabic": "ar",
}


# ===================================================================
# GENERATION FUNCTIONS
# ===================================================================
async def generate_edge_tts(text, voice_name, output_path):
    """Generate audio using edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_path)
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000


def generate_gtts(text, lang_code, output_path):
    """Generate audio using gTTS."""
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tts.save(output_path)
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000


async def generate_for_language(language, texts, output_dir, target_clips=50):
    """Generate fake clips for a single language using best available TTS."""
    lang_dir = os.path.join(output_dir, language, "fake")
    os.makedirs(lang_dir, exist_ok=True)

    manifest_entries = []
    generated = 0
    failed = 0

    # Determine TTS strategy
    use_edge = language in EDGE_TTS_VOICES
    use_gtts = language in GTTS_LANGS

    if not use_edge and not use_gtts:
        print(f"  ⚠ No TTS engine available for {language}")
        return manifest_entries

    texts_to_use = texts[:target_clips + 10]  # buffer for failures

    for i, text in enumerate(texts_to_use):
        if generated >= target_clips:
            break

        # Alternate voices for gender balance
        if use_edge:
            voices = EDGE_TTS_VOICES[language]
            voice = voices[i % len(voices)]
            voice_name = voice["name"]
            gender = voice["gender"]
            engine = "edge-tts"
            ext = "mp3"
            out_path = os.path.join(lang_dir, f"{language}_{i:04d}_{voice_name}.{ext}")

            try:
                success = await generate_edge_tts(text, voice_name, out_path)
                if success:
                    manifest_entries.append({
                        "path": os.path.abspath(out_path),
                        "label": 1,
                        "language": language,
                        "source": f"fake_{language}_{engine}",
                        "tts_engine": engine,
                        "voice": voice_name,
                        "gender": gender,
                        "text": text[:80],
                    })
                    generated += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    edge-tts failed on clip {i}: {e}")
                failed += 1

                # Fallback to gTTS
                if use_gtts:
                    gtts_path = os.path.join(
                        lang_dir, f"{language}_{i:04d}_gtts.mp3")
                    try:
                        success = generate_gtts(
                            text, GTTS_LANGS.get(language, "en"), gtts_path)
                        if success:
                            manifest_entries.append({
                                "path": os.path.abspath(gtts_path),
                                "label": 1,
                                "language": language,
                                "source": f"fake_{language}_gtts",
                                "tts_engine": "gtts",
                                "voice": f"gtts_{GTTS_LANGS.get(language, 'en')}",
                                "gender": "unknown",
                                "text": text[:80],
                            })
                            generated += 1
                            failed -= 1  # recovered
                    except Exception as e2:
                        print(f"    gTTS fallback also failed: {e2}")

        elif use_gtts:
            # Primary: gTTS (for Yoruba)
            lang_code = GTTS_LANGS[language]
            engine = "gtts"
            out_path = os.path.join(lang_dir, f"{language}_{i:04d}_gtts.mp3")

            try:
                success = generate_gtts(text, lang_code, out_path)
                if success:
                    manifest_entries.append({
                        "path": os.path.abspath(out_path),
                        "label": 1,
                        "language": language,
                        "source": f"fake_{language}_{engine}",
                        "tts_engine": engine,
                        "voice": f"gtts_{lang_code}",
                        "gender": "unknown",
                        "text": text[:80],
                    })
                    generated += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    gTTS failed on clip {i}: {e}")
                failed += 1

    print(f"  {language}: generated {generated}/{target_clips} clips "
          f"({failed} failures)")
    return manifest_entries


# ===================================================================
# MAIN
# ===================================================================
async def async_main():
    parser = argparse.ArgumentParser(
        description="Generate TTS fakes for bias audit")
    parser.add_argument("--output-dir", type=str, default="./bias_audit_fakes",
                        help="Output directory for generated audio")
    parser.add_argument("--clips-per-lang", type=int, default=50,
                        help="Target clips per language")
    parser.add_argument("--language", type=str, default=None,
                        help="Generate for single language only")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available edge-tts voices and exit")
    args = parser.parse_args()

    if args.list_voices:
        import edge_tts
        voices = await edge_tts.list_voices()
        # Filter for relevant locales
        relevant = ["en-NG", "ha-NG", "yo-NG", "ig-NG", "pcm-NG",
                     "fr-FR", "ar-SA", "ar-EG"]
        print("Relevant edge-tts voices:")
        for v in voices:
            for locale in relevant:
                if v["Locale"].startswith(locale.split("-")[0]):
                    print(f"  {v['ShortName']:<30} {v['Locale']:<10} "
                          f"{v['Gender']:<8} {v['FriendlyName']}")
                    break
        return

    os.makedirs(args.output_dir, exist_ok=True)

    languages = list(TEXTS.keys())
    if args.language:
        if args.language not in TEXTS:
            print(f"Unknown language: {args.language}")
            print(f"Available: {', '.join(TEXTS.keys())}")
            return
        languages = [args.language]

    print(f"Generating TTS fakes for bias audit")
    print(f"Output: {args.output_dir}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Target: {args.clips_per_lang} clips per language\n")

    all_entries = []
    for lang in languages:
        print(f"\n--- {lang.upper()} ---")
        t0 = time.time()
        entries = await generate_for_language(
            lang, TEXTS[lang], args.output_dir, args.clips_per_lang)
        all_entries.extend(entries)
        print(f"  Time: {time.time()-t0:.1f}s")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "bias_fakes_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total clips: {len(all_entries)}")
    print(f"  Manifest: {manifest_path}\n")

    from collections import Counter
    by_lang = Counter(e["language"] for e in all_entries)
    by_engine = Counter(e["tts_engine"] for e in all_entries)
    by_gender = Counter(e["gender"] for e in all_entries)

    print(f"  By language:")
    for lang, n in sorted(by_lang.items()):
        print(f"    {lang:<12} {n}")
    print(f"\n  By engine:")
    for eng, n in sorted(by_engine.items()):
        print(f"    {eng:<12} {n}")
    print(f"\n  By gender:")
    for g, n in sorted(by_gender.items()):
        print(f"    {g:<12} {n}")

    print(f"\n  Next steps:")
    print(f"  1. Collect real audio for each language into {args.output_dir}/<lang>/real/")
    print(f"  2. Upload the full {args.output_dir}/ to a Kaggle dataset")
    print(f"  3. Run the bias audit evaluation script on Kaggle")
    print(f"{'='*60}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
