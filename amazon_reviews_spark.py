import pyspark
import sys
from pyspark.sql import SparkSession, Row
from textblob import TextBlob
from pyspark.context import SparkContext
from pyspark import SparkConf

files_list_file_name = sys.argv[1]
outputFolderPath = "workspacepython/FinalProject/Customer-Satisfaction-across-Amazon-categories./output2/"



configuration = SparkConf().setMaster("local") \
                            .setAppName("Amazon Reviews Sentiment Analysis") \
                            .set("spark.executor.memory","4g") \
                            .set("sprrk.executor.instances", 1)

sparkcontext = SparkContext(conf = configuration)
spark = SparkSession(sparkcontext)
files_list  = sparkcontext.textFile(files_list_file_name)

amazon_reviews = files_list.map(lambda line: line.split("\t"))
#amazon_reviews.toDF().show()

def getPolarityValue(x):
    if float(x) == 0.0: return 0
    elif float(x) > 0.0: return 1
    else: return -1

#RDD with customer_id , product_id, product_title,  product_category, polarity
amazon_reviews_RDD = amazon_reviews.map(lambda x: (str(x[1]), str(x[3]), str(x[5]), str(x[6]), getPolarityValue((TextBlob(str(x[13])).sentiment.polarity))))
amazon_reviews_RDD.toDF().show()


## to get negative, neutral and positive for each category and save it to  a text file.
amazon_reviews_with_polarity_reduced = amazon_reviews_RDD.map(lambda x: (x[3], (0,0,1) if x[4]==1 else (0,1,0) if x[4] == 0 else (1,0,0)))\
                                                         .reduceByKey(lambda x,y : (x[0]+y[0], x[1]+y[1] , x[2]+y[2]))\
                                                         .map(lambda x : str(x[0]) + "\t" + str(x[1][0]) + "\t" + str(x[1][1]) + "\t" + str(x[1][2]) + "\t" + str( int( int(x[1][0]) + int(x[1][1]) + int(x[1][2]) ) ))
# str(int(int(x[1][0])+int(x[1][1])+int(x[1][2])))
#amazon_reviews_with_polarity_reduced.toDF().show()
amazon_reviews_with_polarity_reduced.coalesce(1).saveAsTextFile(outputFolderPath+ "reviews_satisfaction_per_category")

## products sorted based on positive rating.
top_products_with_reviews_counted = amazon_reviews.map(lambda x: (str(x[1]), str(x[3]), str(x[5]), str(x[6]), getPolarityValue((TextBlob(str(x[13])).sentiment.polarity))))\
                                                .map(lambda x: (x[1], (0,0,1, x[2]) if x[4]==1 else (0,1,0, x[2]) if x[4] == 0 else (1,0,0, x[2])))\
                                                .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2] + y[2], x[3]))
                                                
# ## products sorted based on positive rating.
top_products_with_positive_reviews = top_products_with_reviews_counted.sortBy(lambda keyValue : -keyValue[1][2]).map(lambda x: str(x[0]) + "\t" + str(x[1][0]) + "\t" + str(x[1][1]) + "\t"+ str(x[1][2]) + "\t"+ str(x[1][3]))
top_products_with_positive_reviews.coalesce(1).saveAsTextFile(outputFolderPath+ "top_products_with_postive_record")

# products sorted baed on negative rating.
top_products_with_negative_reviews = top_products_with_reviews_counted.sortBy(lambda keyValue : -keyValue[1][0]).map(lambda x: str(x[0]) + "\t" + str(x[1][0]) + "\t" + str(x[1][1]) + "\t"+ str(x[1][2]) + "\t"+ str(x[1][3]))
top_products_with_negative_reviews.coalesce(1).saveAsTextFile(outputFolderPath+ "top_products_with_negative_record")

# ## Customer with More number of reviews written.
top_customer_reviewers = amazon_reviews.map(lambda x: (str(x[1]), str(x[3]), str(x[5]), str(x[6]), getPolarityValue((TextBlob(str(x[13])).sentiment.polarity))))\
                                                .map(lambda x: (x[0], (0,0,1, x[3]) if x[4]==1 else (0,1,0, x[3]) if x[4] == 0 else (1,0,0, x[3])))\
                                                .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2] + y[2], x[3]))

top_customer_reviewers_positive = top_customer_reviewers.sortBy(lambda keyValue : -keyValue[1][2]).map(lambda x: str(x[0]) + "\t" + str(x[1][0]) + "\t" + str(x[1][1]) + "\t"+ str(x[1][2]) + "\t"+ str(x[1][3]))
#top_customer_reviewers_positive.toDF().show()
top_customer_reviewers_positive.coalesce(1).saveAsTextFile(outputFolderPath+ "top_positive_reviewers")
# #RDD with customer_id , product_id, product_title,  product_category, polarity
# amazon_reviews_with_product_id = amazon_reviews.map(lambda x: (str(x[1]), str(x[3]), str(x[5]), str(x[6]), getPolarityValue((TextBlob(str(x[13])).sentiment.polarity))))\
#                                                 .map(lambda x: (x[1], (0,0,1) if x[4]==1 else (0,1,0) if x[4] == 0 else (1,0,0)))\
#                                                 .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1] , x[2]+y[2]))
# amazon_reviews_with_product_id.toDF().show()

# top_20_positive_products = amazon_reviews_with_product_id.sortBy(lambda keyValue : -keyValue[1][2])
# top_20_positive_products.coalesce(1).saveAsTextFile("workspacepython/FinalProject/testoutput_top_20_positive_review_products")





#amazon_review_RDD1 = amazon_reviews.map(lambda x: (str(x[6]), str(x[7]), str(x[8]), str(x[13]), (TextBlob(str(x[13])).sentiment.polarity)))
#amazon_review_RDD1.toDF().show()

# RDD with product_cateory, star_rating, helpful_votes, review_body, polarity
# amazon_review_RDD1 = amazon_reviews.map(lambda x: (str(x[6]), str(x[7]), str(x[8]), str(x[13]), getPolarityValue((TextBlob(str(x[13])).sentiment.polarity))))
# amazon_review_RDD1.toDF().show()


# def getTupleValue(x):
#     if x == 0:
#         return tuple([0,1,0])
#     elif x == 1:
#         return tuple([0,0,1])
#     else:
#         return tuple([1,0,0])
#amazon_review_RDD2 = amazon_review_RDD1.map(lambda x: (x[0], getTupleValue(x[4])))

#Rdd with porduct_cateogry and then (negative, neutral, positive) count.
# amazon_review_RDD2 = amazon_review_RDD1.map(lambda x: (x[0], (0,0,1) if x[4]==1 else (0,1,0) if x[4] == 0 else (1,0,0)))
# amazon_review_RDD2.toDF().show()

# # reduce the above RDD to get the count of the values for negative, neutral and positive.
# amazon_review_RDD3 = amazon_review_RDD2.reduceByKey(lambda x,y : (x[0]+y[0], x[1]+y[1] , x[2]+y[2]))

# #RDD to print the value to a text file as tsv
# amazon_review_RDD4 = amazon_review_RDD3.map(lambda x : str(x[0]) + "\t" + str(x[1][0]) + "\t" + str(x[1][1]) + "\t" + str(x[1][2]))
# #amazon_review_RDD4.toDF().show()

# amazon_review_RDD4.coalesce(1).saveAsTextFile("workspacepython/FinalProject/testoutput9")
# print("##############################################")


#RDD with product_category and then helful_votes * polarity
# amazon_reviews_category_reduced_RDD = amazon_review_RDD1.map(lambda x : (x[0], x[4]) if(int(x[2]) == 0) else (x[0], x[4]*int(x[2]))).reduceByKey(lambda a,b : a+b)

#RDD to print the value to a text file as tsv
#mazon_reviews_reduced_rdd_2 = amazon_reviews_category_reduced_rdd.map(lambda x: str(x[0]) + "\t" + str(x[1]))


#amazon_reviews_category_reduced_rdd.saveAsTextFile("workspacepython/FinalProject/testoutput1")
#amazon_reviews_category_reduced_rdd.coalesce(1).saveAsTextFile("workspacepython/FinalProject/testoutput3")
#print("#################################")
#print(k)

#print(amazon_reviews.take(5))

