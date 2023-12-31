# securityautofill-model

This project takes the messages.csv file from our securityautofill-trainingdata package and trains an NLP model to identify security codes.

## Making the data useful

To turn our messages.csv into a useful train.csv and eval.csv file, we need to label our data, which is a massive pain in the arse.
I spent about half an hour looking into automatic options and tools like Doccano, Prodigy, and Label Studio, before finding them all (to varying degrees) a massive waste of time. In the end I just cranked out Numbers and went hell-for-leather on various filters on the data to tease out interesting examples – specifically OTPs in different languages, and OTPs that don't contain obvious words like "code" or "security" or "OTP" etc.

This resulted in about ~600 "codes" and ~1000 "not codes" labelled in `messages-labelled.csv`, and around ~10,000 unlabelled records in `messages-unlabelled.csv`.
I then split the labelled training data 70%/30% into `messages-labelled-s.csv` and `messages-labelled-s-eval.csv` respectively, so that there could be some evaluation in the process of the model's training.

## Training the model

I took two different approaches to training – one supervised, and one semi-supervised.

#### SetFit (supervised)

The supervised approach uses SetFit to generate a model from our labelled data.
SetFit is designed for few-shot learning with minimal classified examples, and I've heard great things about SetFit, but I also felt like it was a shame to waste the ~10k other records we had, so I also went for a second approach in parallel...

#### Trainer (semi-supervised)

The semi-supervised approach uses the Trainer class from HuggingFace's Transformers library to train a model on our labelled data, and then fine-tune it on our unlabelled data.
That way, we don't waste any data. It's self-reinforcing (i.e. training itself on its own predictions) which will never be as good as properly labelled data, but given this data is entirely made up of random public phone numbers which mostly receive dodgy texts and OTP codes, the types of messages are the same as its training data, meaning its predictions are pretty likely to be correct – so this semi-supervised approach should reinforce its predictions to hold up to scrutiny against real-world text messages.

## Which is best

No idea yet. Will get back to you on that one.
The plan is to run the `securityautofill-trainingdata` package again and do some more manual labelling, and evaluate both models against that data.
For now, I'm just presuming that the semi-supervised approach will do a better job, because it's using more data.
