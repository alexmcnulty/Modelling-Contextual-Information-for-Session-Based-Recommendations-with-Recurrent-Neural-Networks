
import operator
import pickle


# functions


def get_keywords(row, publisher):
    """

    :param row: Pass in a row
    :param publisher: define whether its a merchant or publisher
    :return: the keywords as the new rdd
    """
    if publisher == True:
        return row['pu_keywords']
    else:
        return row['merc_url_kw']


def filter_kw(row, embedd_dict):
    """
    Reduce the size of the embedding
    :param row: list of keywords that appear in merchant and publisher urls
    :param embedd_dict: gensim embedding as dictionary
    :return: a new dictionary with only the words that appear in the url keywords
    """
    dict = {}
    for i in row:
        try:
            word = embedd_dict[i]
            dict[i] = word
        except:
            pass
    return dict

def get_merchants(row):

    return row['merchantDomain']

def get_publishers(row):

    return row['domainId']
################################################################

# sequence functions

def pub_imp_cols(row):
    """Parses page impressions to tensorflow compatible format.

    """
    return (row['guid'], ((row['timestamp'], row['domainId'], row['pu_keywords'], row['year'], row['month'],row['day'])))


def get_click_imp(row):
    """Parses page impressions to tensorflow compatible format.

    (guid, (2016-01-01, domain_id, pub_url, pub_imp))
    """
    return (
    row['guid'], ((row['timestamp'], row['merchantDomain'], row['merc_url_kw'], row['year'], row['month'], row['day'])))


def aggregate_ts_and_domain_id(a, b):
    """Sorts the (ts, domain_id,  pub_url, pub_imp)) tuples by ts."""
    # basically when we add a and b we are actually adding tuples
    # maybe we need to do a map first to get in the correct type as done in the dummy code
    raw = operator.add(a, b)
    return raw

def sort_by_ts(row):
    """Sorts the tuple of tuples by ts

    in ((123, 'hello'), (01, 'world'))
    out ((01, 'world'), (123, 'hello'))
    """
    return sorted(row, key=get_key)


def get_key(item):
    return item[0]


def get_tuples(row):
    "organise by guid and tuples of data"
    # here can change to have a list.t
    return (row
            .map(lambda (guid, data): (guid,[data]))
            .reduceByKey(lambda a, b: aggregate_ts_and_domain_id(a, b)))


def filter_low_counts(row, domain,count):
    ids = [x[1] for x in row]
    for x in row:
        if domain[x[1]] <= count:
            return [None]
    return row


def dwell_time(row):
    time_stamps = [x[0] for x in row]
    newSeq = []
    dts = []
    for i in range(len(time_stamps) - 1):
        dt = time_stamps[i+1] - time_stamps[i]
        dts.append(dt)
        new_tuple = row[i] + (dt,)
        newSeq.append(new_tuple)
    final_ts =  np.mean(dts)
    final_entry = row[-1] + (final_ts,)
    newSeq.append(final_entry)
    return newSeq


def generate_sessions(row, dwelling):
    newseq = []
    session = []
    count = 0
    dwelltimes = [x[-1] for x in row]
    for elm in dwelltimes:
        if elm <= dwelling:
            session.append(row[count])
            count+=1
            if count == len(row):
                newseq.append(session)
        else:
            session.append(row[count])
            newseq.append(session)
            count += 1
            session = []

    return newseq

sqlContext = SQLContext(sc)

page_imp = sqlContext.read.parquet('s3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=02/profiles.parquet/*',
                                            's3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=03/profiles.parquet/*',
                                            's3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=04/profiles.parquet/*',
                                            's3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=05/profiles.parquet/*',
                                            's3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=06/profiles.parquet/*',
                                            's3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=07/profiles.parquet/*',
                                            's3n://audience-data-store/production/enrichment/page_impressions/year=2017/month=06/day=08/profiles.parquet/*')

clicks = sqlContext.read.parquet('s3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=02/profiles.parquet/*'
                                          's3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=03/profiles.parquet/*',
                                          's3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=04/profiles.parquet/*',
                                          's3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=05/profiles.parquet/*',
                                          's3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=06/profiles.parquet/*',
                                          's3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=07/profiles.parquet/*',
                                          's3n://audience-data-store/production/enrichment/clicks_a/year=2017/month=06/day=08/profiles.parquet/*')

# Need to get the dictionary of key words from gensim of the merchant urls and the puburls

# first turn to rdd and grab keywords
pi_rdd = page_imp.rdd.repartition(2000).cache()
clicks_rdd = clicks.rdd.repartition(2000).cache()
pi_rdd.count()
clicks_rdd.count()


# Get keywords
pi_kw = pi_rdd.map(lambda a: get_keywords(a,True))
clicks_kw = clicks_rdd.map(lambda a: get_keywords(a,False))

# need to flatmap and find distinct words
pi_kw = pi_kw.flatMap(lambda a:a).map(lambda a: (a,1)).reduceByKey(lambda a,b: a+b)
clicks_kw = clicks_kw.flatMap(lambda a:a).map(lambda a: (a,1)).reduceByKey(lambda a,b: a+b)

# join the kw together and take the keys of the converted dictionary
joint_kw = pi_kw.fullOuterJoin(clicks_kw)
key_words = joint_kw.collectAsMap().keys()

# Get the gensim converted to a dictionary that can be used to extract the keywords
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# convert embedding from gensim object to dict
d = {}
s = model.__dict__['syn0']
v = model.__dict__['vocab']
for w,t in v.items():
    index = t.__dict__['index']
    d[w] =  s[index]

# Filter out the words that don't appear in the gensim dictionary
key_words_dictionary = filter_kw(key_words, d)

# save the dictionary
with open("key_words.pkl", "wb") as f:
    pickle.dump(key_words_dictionary, f)

######################################################################

# Grab unique merchants to be used later on for prior on unseen merchants
unique_pubs = pi_rdd.map(get_publishers).map(lambda a: (a,1)).reduceByKey(lambda a,b: a+b)
unique_pubs = unique_pubs.collectAsMap().keys()

with open("publisher_list.pkl", "wb") as f:
    pickle.dump(unique_pubs, f)

unique_merchants = clicks_rdd.map(get_merchants).map(lambda a: (a,1)).reduceByKey(lambda a,b: a+b)
unique_merchants = unique_merchants.collectAsMap().keys()

with open("merchant_list.pkl", "wb") as f:
    pickle.dump(unique_merchants, f)
# Next need to grab sequences of users

# first get the columns were interested in
filtered_pi = pi_rdd.map(pub_imp_cols).cache()
filtered_pi.count()

filtered_clicks = clicks_rdd.map(get_click_imp).cache()
filtered_clicks.count()

# next collect per user
filtered_pi = get_tuples(filtered_pi).cache()
filtered_pi.count()

filtered_clicks = get_tuples(filtered_clicks).cache()
filtered_clicks.count()

# Need the data so that we have at least one merchant per user this implies we should use a right outer join
joinedData = filtered_pi.rightOuterJoin(filtered_clicks).cache()
joinedData.count()
data = joinedData.filter(lambda a : a[1][0] != None)

# concatonate the merchant and publisher tuples
data = data.map(lambda a: (a[0], list(a[1][0])+a[1][1]))

# sort by ts
data = data.map(lambda a: (a[0], sort_by_ts(a[1])))
data.count()
# Not filtering out low counts, try later if doesn't work too well

#########################################
# filter out low counts
domains = data.map(lambda a: [x[1] for x in a[1]])
domains = domains.flatMap(lambda a: a).map(lambda a: (a,1))
domains = domains.reduceByKey(lambda a,b: a+b)
domains_dict = domains.collectAsMap()

domains_reduced = domains.filter(lambda a: a[1]>2 ).collectAsMap().keys()

with open("all_domains.pkl", "wb") as f:
    pickle.dump(domains_reduced, f)

filtered_data = data.map(lambda a: (a[0],filter_low_counts(a[1],domains_dict,2)))

filtered_data = filtered_data.filter(lambda a: a[1] != [None])

data = filtered_data

# append dwell time
dwell_data = data.map(lambda a: (a[0], dwell_time(a[1])))

dwell_data = dwell_data.map(lambda a: (a[0],generate_sessions(a[1],1800)))

# get the sessions of length greater than 2
seq_data = dwell_data.flatMap(lambda a:a[1])

seq_lens = seq_data.filter(lambda a: len(a) >= 3)

seq_lens = seq_lens.filter(lambda a: len(a)<=20)

# Save the full sequences without dwell time reducing session
full_seqs = data.coalesce(1)
full_seqs.saveAsTextFile("s3n://audience-data-store-qa/alex/keywords/one_week/full_seqs")

# save as dwell time sessions
sessions = seq_lens.coalesce(1)
sessions.saveAsTextFile("s3n://audience-data-store-qa/alex/keywords/one_week/sessions_seqs_final")

##################

# get the domain embedding using sparks skip-gram model

# need to convert to string for word2vec in spark
stop_vec = ['UNK']*3
data2 = seq_lens.map(lambda a : a + stop_vec )

# turn to string to be read into Word2Vec
data2 = data2.map(lambda x: [str(i) for i in x])

from pyspark.mllib.feature import Word2Vec

k = 50        # vector dimensionality

model = Word2Vec().setVectorSize(k).setMinCount(1).setNumIterations(50).setWindowSize(3).fit(data2)

vectors = model.getVectors()
vectors = dict(vectors)

# Need to convert the py4j lists to python lists.
vecs = {x: list(vectors[x]) for x in vectors}

with open('oneweek_embedding.pkl', 'wb') as handle:
    pickle.dump(vecs, handle)

# write the embedding
s3tools.writepickletoS3(vecs,'s3://audience-data-store-qa/alex/data/one_month/word2vec/embedding.pkl')