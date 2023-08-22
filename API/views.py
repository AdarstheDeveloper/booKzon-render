from django.shortcuts import render
from rest_framework.response import Response
from .models import Books
from .serializer import LeadSerializer
from rest_framework.decorators import api_view

import json
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

vectorizer = TfidfVectorizer()

books_json = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data/books.json')
    
titles = pd.read_json(books_json)
titles["ratings"] = pd.to_numeric(titles["ratings"])
titles["mod_title"] = titles["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
titles["mod_title"] = titles["mod_title"].str.lower()
titles["mod_title"] = titles["mod_title"].str.replace("\s+", " ", regex=True)
titles = titles[titles["mod_title"].str.len() > 0]

tfidf = vectorizer.fit_transform(titles["mod_title"])


@api_view()
def search_book (request, query='') :
        Books.objects.all().delete()
        processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
        query_vec = vectorizer.transform([processed])
        similarity = cosine_similarity(query_vec, tfidf).flatten()
        indices = np.argpartition(similarity, -10)[-12:]
        results = titles.iloc[indices]
        results = results.sort_values("ratings", ascending=False)
        i = 0
        while(i < 12) :
            book_id = (results.iloc[i]['book_id'])
            title = (results.iloc[i]['title'])
            url = (results.iloc[i]['url'])
            cover_image = (results.iloc[i]['cover_image'])
            i = i + 1
            Books(id=i,title=title, book_id=book_id, url=url, cover_image=cover_image).save()
        books = Books.objects.all()
        serializer = LeadSerializer(books, many=True)
        return Response(serializer.data, template_name=None)

