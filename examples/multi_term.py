## multi term keyword example

from mrakun import RakunDetector
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

blob_of_text = "Brexit (/ˈbrɛksɪt, ˈbrɛɡzɪt/;[1] a portmanteau of \"British\" and \"exit\") is the scheduled withdrawal of the United Kingdom (UK) from the European Union (EU). Following a June 2016 referendum, in which 51.9% voted to leave, the UK government formally announced the country's withdrawal in March 2017, starting a two-year process that was due to conclude with the UK withdrawing on 29 March 2019. As the UK parliament thrice voted against the negotiated withdrawal agreement, that deadline has been extended twice, and is currently 31 October 2019.[2][3] An Act of Parliament requires the government to seek a third extension if no agreement is reached before 19 October. Withdrawal is advocated by Eurosceptics and opposed by pro-Europeanists, both of whom span the political spectrum. The UK joined the European Communities (EC) in 1973, with continued membership endorsed in a 1975 referendum. In the 1970s and 1980s, withdrawal from the EC was advocated mainly by the political left, e.g. in the Labour Party's 1983 election manifesto. From the 1990s, the eurosceptic wing of the Conservative Party grew, and led a rebellion over ratification of the 1992 Maastricht Treaty that established the EU. In parallel with the UK Independence Party (UKIP), and the cross-party People's Pledge campaign, it pressured Conservative Prime Minister David Cameron to hold a referendum on continued EU membership. Cameron, who had campaigned to remain, resigned after the result and was succeeded by Theresa May. On 29 March 2017, the UK government invoked Article 50 of the Treaty on European Union, formally starting the withdrawal. May called a snap general election in June 2017, which resulted in a Conservative minority government supported by the Democratic Unionist Party. UK–EU withdrawal negotiations began later that month. The UK negotiated to leave the EU customs union and single market. This resulted in the November 2018 withdrawal agreement, but the UK parliament voted against ratifying it three times. The Labour Party wanted any agreement to maintain a customs union, while many Conservatives opposed the agreement's financial settlement on the UK's share of EU financial obligations, as well as the Irish backstop designed to prevent border controls in Ireland. The Liberal Democrats, Scottish National Party and others seek to reverse Brexit through a second referendum. The EU has declined a re-negotiation that omits the backstop. In March 2019, the UK parliament voted for May to ask the EU to delay Brexit until October. Having failed to pass her agreement, May resigned as Prime Minister in July and was succeeded by Boris Johnson. He sought to replace parts of the agreement and vowed to leave the EU by the new deadline, with or without an agreement."

hyperparameters = {
    "distance_threshold": 4,
    "distance_method": "editdistance",
    "num_keywords": 10,
    "pair_diff_length": 3,
    "stopwords": stopwords.words('english'),
    "bigram_count_threshold": 0,
    "lemmatizer": lemmatizer,
    "num_tokens": [2]
}

keyword_detector = RakunDetector(hyperparameters)
keywords = keyword_detector.find_keywords(blob_of_text, input_type="text")
print(keywords)
