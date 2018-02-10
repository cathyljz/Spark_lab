
# coding: utf-8

# # CS5344: Spark lab 

# Author: Li Jiazhe (A0176576M)

# ## Stage 1

# • Input: a set of files under a directory
# 
# • Compute frequency of every word in a document

# In[1]:


from pyspark import SparkConf, SparkContext
import re
import math


# In[2]:


conf = (SparkConf()
         .setMaster("local")
         .setAppName("WordCounter")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)


# In[3]:


f = open("/home/spark/Downloads/lab1/stopwords.txt","r",encoding="utf-8")
lines = f.readlines()
stopwords = [x.strip() for x in lines] 


# In[4]:


if __name__ == "__main__":

    def process_words(context): # leave out numbers and punctuations and split words
        path = context[0] 
        prewords = re.sub('[^a-z]+',' ',context[1].lower()).split() 
        words = [word for word in prewords if word not in stopwords]
        file = path.split("/")[-1] # get the file name from path
      
        # appending the file name
        list = [x + '@' + file for x in words]
        return list

    text_files = sc.wholeTextFiles("/home/spark/Downloads/lab1/datafiles/*")
    word_doc_count = text_files.flatMap(process_words)                       .map(lambda w: (w,1))                        .reduceByKey(lambda x,y: x+y)
                
    docs_count = len(text_files.collect()) # total number of files


# ## Stage2

# • Input: Stage 1 output
# 
# • Compute TF-IDF of every word w.r.t a document

# In[5]:


if __name__ == "__main__":
    
    def word_docid_count(val): # input word_doc_count (k-v pair:"word@doc:count"), output k-v pair:"word:docid@count"
        word_docid, count = val 
        word, docid = word_docid.split('@')  
        return (word, '{0}@{1}'.format(docid, count))

    def word_docid_count_docswithword(val): # input k-v pair:"word:docid@count", output k-v pair:"word@doc:count@counter
        word = val[0]
        docid_count_pairs = []
        counter = 0
        for docid_count in val[1]: 
            counter = counter + 1 # after group by key, count the number of appearance of the key so that  
                                  # we get the number of docs that contains the key word.
            docid, count = docid_count.split('@')
            docid_count_pairs.append((docid, count)) # further split the value to a list with pairs for calculation
        
        result = []
        for (docid, count) in docid_count_pairs:
            word_docid = '{0}@{1}'.format(word, docid)
            count_docswithword = '{0}@{1}'.format(count, counter)
            result.append((word_docid, count_docswithword)) 
        return result

    def count_tf_idf(val): # input k-v pair:"word@doc:count@counter", output k-v pair:"word@doc:tfidf"
        word_docid, count_docswithword = val
        fdt, dft = [int(x) for x in count_docswithword.split('@')]
        # fdt: number of the word appears in the doc; dft: counter
        return (word_docid, (1 + math.log(fdt)) * math.log(docs_count / dft))
    
    tfidf_raw = word_doc_count.map(word_docid_count)                              .groupByKey()                              .flatMap(word_docid_count_docswithword)                              .map(count_tf_idf)


# ## Stage 3

# • Input: Stage 2 output
# 
# • Compute normalized TF-IDF of every word w.r.t. a document

# In[6]:


if __name__ == "__main__":
    
    def docid_word_tfidf(val): # input tfidf (k-v pair:"word@doc:tfidf"), output k-v pair:"docid:word@tfidf"
        word_docid, tfidf = val # word_docid:"word@doc
        word, docid = word_docid.split('@')  
        return (docid, '{0}@{1}'.format(word, tfidf))

    def docid_word_sos_tfidf(val): # input k-v pair:"docid:word@tfidf", output k-v pair:"word@doc:tfidf@sos_tfidf
        docid = val[0]
        word_tfidf_pairs = []
        sos_tfidf = 0 # sum of square tfidf
        for word_tfidf in val[1]: 
            word, tfidf = word_tfidf.split('@')
            sos_tfidf = sos_tfidf + float(tfidf) ** 2
            word_tfidf_pairs.append((word, tfidf)) 
        sqrt_sos_tfidf = math.sqrt(sos_tfidf)
        
        result = []
        for (word, tfidf) in word_tfidf_pairs:
            word_docid = '{0}@{1}'.format(word, docid)
            tfidf_sqrt_sos_tfidf = '{0}@{1}'.format(tfidf, sqrt_sos_tfidf)
            result.append((word_docid, tfidf_sqrt_sos_tfidf)) 
        return result

    def norm_tf_idf(val): # input k-v pair:"word@doc:tfidf@sos_tfidf", output k-v pair:"word@doc:n_tfidf
        word_docid, tfidf_sqrt_sos_tfidf = val
        tfidf, sqrt_sos_tfidf = [float(x) for x in tfidf_sqrt_sos_tfidf.split('@')]
        return (word_docid, tfidf/sqrt_sos_tfidf)

    norm_tfidf = tfidf_raw.map(docid_word_tfidf)                          .groupByKey()                          .flatMap(docid_word_sos_tfidf)                          .map(norm_tf_idf)


# ## Stage 4

# • Input: Stage 4 output and a query file query.txt
# 
# • Compute relevance of every document w.r.t a query

# In[7]:


q = open("/home/spark/Downloads/lab1/query.txt","r",encoding="utf-8")
query = q.read().split(" ")


# In[8]:


if __name__ == "__main__":
    
    def docid_n_tfidf_words(val): # input k-v pair:"word@doc:n_tfidf", output k-v pair:"doc@n_tfidf:word"
        word_docid, n_tfidf = val
        word, docid = word_docid.split('@') 
        return ('{0}@{1}'.format(docid, n_tfidf), word)
    
    def docid_word_norm_tfidf(val): # input k-v pair:"doc@n_tfidf:word", output k-v pair:"docid:n_tfidf"
        docid_n_tfidf, word = val 
        docid, n_tfidf = docid_n_tfidf.split('@')  
        return (docid, float(n_tfidf))
    
    relevance = norm_tfidf.map(docid_n_tfidf_words)                          .filter(lambda x: x[1] in query)                          .map(docid_word_norm_tfidf)                          .reduceByKey(lambda x,y: x+y)


# ## Stage 5

# • Input: Stage 4 output
# 
# • Sort documents by their relevance to the query in descending order
# 
# • Output the top-k documents

# In[9]:


sorted_relevance = relevance.sortBy(lambda x: x[1], ascending=False)
                    
print(sorted_relevance.collect())


# As we can see, the sequence of relevance to the query is f4 > f8 > f1 > f3 > f2 > f5 > f10.
