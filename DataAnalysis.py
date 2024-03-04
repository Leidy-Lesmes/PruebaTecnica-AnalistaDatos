import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

file_path_jobs = "data/large-jobs.csv"
file_path_employments_types = "data/large-employments_types.csv"
file_path_skills = "data/large-skills.csv"
file_path_multilocations = "data/large-multilocations.csv"

column_names = ["title", "street", "city", "country_code", "address_text", "marker_icon", 
                "workplace_type", "company_name", "company_url", "company_size", 
                "experience_level", "published_at", "remote_interview", 
                "open_to_hire_ukrainians", "id", "display_offer"]

df_jobs = pd.read_csv(file_path_jobs, delimiter=';', names=column_names)
df_employment_types = pd.read_csv(file_path_employments_types, delimiter=";", usecols=[1, 3, 4], header=None)
df_skills = pd.read_csv(file_path_skills, delimiter=";", header=None, names=["skill", "experience_level", "id"])
df_multilocations = pd.read_csv(file_path_multilocations, delimiter=";", header=None, names=["city", "street", "id"])

df_employment_types.columns = ['id', 'salary_from', 'salary_to']

df_merged = pd.merge(df_jobs, df_employment_types, on='id', how='left')


total_offers_loaded = len(df_jobs)

df_jobs_sorted = df_jobs.sort_values(by='published_at', ascending=True)
first_offers = df_jobs_sorted.head(3)
last_offers = df_jobs_sorted.tail(4)

print("")
print("Total de ofertas de trabajo publicadas cargadas:", total_offers_loaded)
print("\nInformación de las tres primeras ofertas de trabajo publicadas:")
print(tabulate(first_offers[['published_at', 'title', 'company_name', 'experience_level', 'country_code', 'city']], headers='keys', tablefmt='pretty'))
print("\nInformación de las tres ultimas ofertas de trabajo publicadas:")
print(tabulate(last_offers[['published_at', 'title', 'company_name', 'experience_level', 'country_code', 'city']], headers='keys', tablefmt='pretty'))


def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

execution_times = []

print("")
print("REQUERIMIENTO #1")
print("")

def list_latest_offers(offers_number, country_code, expertise_level):
    leaked_offers = df_jobs[(df_jobs['country_code'] == country_code) & (df_jobs['experience_level'] == expertise_level)]
    leaked_offers = leaked_offers.sort_values(by='published_at', ascending=False)
    latest_offers = leaked_offers.head(offers_number)
    return latest_offers

### Definicion de valores de los parametros
offers_number = 5
country_code = 'PL'
expertise_level = 'junior'
###

latest_offers = list_latest_offers(offers_number, country_code, expertise_level)

columns_to_show = ['published_at', 'title', 'company_name', 'experience_level', 'country_code', 'city', 'company_size', 'workplace_type', 'open_to_hire_ukrainians']

print(tabulate(latest_offers[columns_to_show], headers='keys', tablefmt='pretty'))

print("********************************************************")
result, execution_time = measure_execution_time(list_latest_offers, offers_number, country_code, expertise_level)
execution_times.append(execution_time)
print("Tiempo de ejecución - Requerimiento 1:", execution_time, "segundos")
print("********************************************************")


print("")
print("REQUERIMIENTO #2")
print("")

def get_company_offers_by_date_range(company_name, start_date, end_date):
    start_date = pd.to_datetime(start_date, format='%Y/%m/%d')
    end_date = pd.to_datetime(end_date, format='%Y/%m/%d')
    
    if not pd.api.types.is_datetime64_any_dtype(df_jobs['published_at']):
        df_jobs['published_at'] = pd.to_datetime(df_jobs['published_at'].str[:10], format='%Y-%m-%d', errors='coerce')
    
    company_offers = df_jobs[(df_jobs['company_name'] == company_name) & 
                             (df_jobs['published_at'].between(start_date, end_date))]
    return company_offers


def analyze_company_offers(company_name, start_date, end_date):
    company_offers = get_company_offers_by_date_range(company_name, start_date, end_date)
    
    total_offers = len(company_offers)
    
    junior_offers = company_offers[company_offers['experience_level'] == 'junior']
    mid_offers = company_offers[company_offers['experience_level'] == 'mid']
    senior_offers = company_offers[company_offers['experience_level'] == 'senior']
    
    total_junior_offers = len(junior_offers)
    total_mid_offers = len(mid_offers)
    total_senior_offers = len(senior_offers)
    
    sorted_company_offers = company_offers.sort_values(by=['published_at', 'country_code'])
    
    return {
        'total_offers': total_offers,
        'total_junior_offers': total_junior_offers,
        'total_mid_offers': total_mid_offers,
        'total_senior_offers': total_senior_offers,
        'sorted_offers': sorted_company_offers
    }

### Definicion de valores de los parametros
start_date = '2022/04/10'
end_date = '2023/09/01'
company_name = 'GMS'
###

result = analyze_company_offers(company_name, start_date, end_date)

print("Número total de ofertas:", result['total_offers'])
print("Número total de ofertas Junior:", result['total_junior_offers'])
print("Número total de ofertas Mid:", result['total_mid_offers'])
print("Número total de ofertas Senior:", result['total_senior_offers'])

print("\nOfertas de la empresa", company_name, "entre", start_date, "y", end_date, ":")
print(tabulate(result['sorted_offers'][['published_at', 'title', 'experience_level', 'city', 'country_code', 'company_size', 'workplace_type', 'open_to_hire_ukrainians']], headers='keys', tablefmt='pretty'))

print("********************************************************")
execution_times.append(execution_time)
result, execution_time = measure_execution_time(analyze_company_offers, company_name, start_date, end_date)
print("Tiempo de ejecución - Requerimiento 2:", execution_time, "segundos")
print("********************************************************")

print("")
print("REQUERIMIENTO #3")
print("")

def get_job_offers_by_country_and_date_range(country_code, start_date, end_date):
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    
    country_offers = df_jobs[(df_jobs['country_code'] == country_code) & 
                             (df_jobs['published_at'].between(start_date, end_date))]
    return country_offers


def analyze_country_job_offers(country_code, start_date, end_date):
    country_offers = get_job_offers_by_country_and_date_range(country_code, start_date, end_date)
    
    total_offers = len(country_offers)
    total_companies = len(country_offers['company_name'].unique())
    total_cities = len(country_offers['city'].unique())

    city_max_offers = country_offers['city'].value_counts().idxmax()
    city_max_offers_count = country_offers['city'].value_counts().max()
    
    city_min_offers = country_offers['city'].value_counts().idxmin()
    city_min_offers_count = country_offers['city'].value_counts().min()
    
    sorted_country_offers = country_offers.sort_values(by=['published_at', 'company_name'])
    
    return {
        'total_offers': total_offers,
        'total_companies': total_companies,
        'total_cities': total_cities,
        'city_max_offers': {
            'city': city_max_offers,
            'count': city_max_offers_count
        },
        'city_min_offers': {
            'city': city_min_offers,
            'count': city_min_offers_count
        },
        'sorted_offers': sorted_country_offers
    }

### Definicion de valores de los parametros
country_code = 'VN'
start_date = '2022-04-10'
end_date = '2023-09-01'
###

result = analyze_country_job_offers(country_code, start_date, end_date)

print("Total de ofertas en el país:", result['total_offers'])
print("Total de empresas que publicaron ofertas en el país:", result['total_companies'])
print("Número total de ciudades del país con ofertas:", result['total_cities'])
print("Ciudad con más ofertas:", result['city_max_offers']['city'], "- Total de ofertas:", result['city_max_offers']['count'])
print("Ciudad con menos ofertas:", result['city_min_offers']['city'], "- Total de ofertas:", result['city_min_offers']['count'])

print("\nOfertas publicadas en", country_code, "entre", start_date, "y", end_date, ":")
columns_to_print = ['published_at', 'title', 'experience_level', 'company_name', 'city', 'workplace_type', 'open_to_hire_ukrainians']
if 'remote' in result['sorted_offers']:
    columns_to_print.append('remote')

print(tabulate(result['sorted_offers'][columns_to_print], headers='keys', tablefmt='pretty'))

print("********************************************************")
result, execution_time = measure_execution_time(analyze_country_job_offers, country_code, start_date, end_date)
execution_times.append(execution_time)
print("Tiempo de ejecución - Requerimiento 3:", execution_time, "segundos")
print("********************************************************")

print("")
print("REQUERIMIENTO #4")
print("")

def classify_cities_with_most_job_offers(city_number_query, country_code=None, expertise_level=None, start_date=None, end_date=None):
    filtered_offers = df_merged.copy()
    if country_code:
        filtered_offers = filtered_offers[filtered_offers['country_code'] == country_code]
    if expertise_level:
        filtered_offers = filtered_offers[filtered_offers['experience_level'] == expertise_level]
    if start_date and end_date:
        filtered_offers = filtered_offers[(filtered_offers['published_at'] >= start_date) & (filtered_offers['published_at'] <= end_date)]

    city_counts = filtered_offers['city'].value_counts().nlargest(city_number_query)
    top_cities = city_counts.index.tolist()

    filtered_offers = filtered_offers[filtered_offers['city'].isin(top_cities)]

    total_cities = len(top_cities)
    total_companies = len(filtered_offers['company_name'].unique())
    total_offers = len(filtered_offers)
    average_salary = filtered_offers[['salary_from', 'salary_to']].mean().mean()

    city_stats = []
    for city, data in filtered_offers.groupby('city'):
        total_offers_city = len(data)
        avg_salary_city = data[['salary_from', 'salary_to']].mean().mean()
        total_companies_city = len(data['company_name'].unique())

        if total_offers_city > 0:
            max_offer = data[['salary_from', 'salary_to']].max(axis=1).max()
            min_offer = data[['salary_from', 'salary_to']].min(axis=1).min()
            max_offer_company = data.loc[data[['salary_from', 'salary_to']].fillna(-np.inf).max(axis=1).idxmax(), 'company_name']

        else:
            max_offer = np.nan
            min_offer = np.nan
            max_offer_company = np.nan

        city_stats.append([city, total_offers_city, avg_salary_city, total_companies_city, max_offer_company, max_offer, min_offer])

    city_stats_df = pd.DataFrame(city_stats, columns=['Ciudad', 'Número de ofertas', 'Promedio del salario ofertado', 'Numero de empresas que ofertaron en la ciudad', 
                                                      'Nombre de la empresa con más ofertas', 'Mejor oferta', 'Peor oferta'])

    city_stats_df = city_stats_df.sort_values(by=['Número de ofertas', 'Ciudad'], ascending=[False, True])

    print("Resultados:")
    print(tabulate([
        ["Total de ciudades", total_cities],
        ["Total de empresas", total_companies],
        ["Total de ofertas publicadas", total_offers],
        ["Promedio del salario ofertado", average_salary],
    ], headers=["Estadística", "Valor"], tablefmt="pretty"))

    print("\nListado de ciudades ordenadas por número de ofertas y nombre de ciudad:")
    print(tabulate(city_stats_df, headers="keys", tablefmt="pretty"))

    return {
        'total_cities': total_cities,
        'total_companies': total_companies,
        'total_offers': total_offers,
        'average_salary': average_salary,
        'cities_sorted': city_stats_df
    }

city_number_query = 5
country_code = 'PL'
expertise_level = 'junior'
start_date = '2022-01-01'
end_date = '2022-12-31'
results = classify_cities_with_most_job_offers(city_number_query, country_code, expertise_level, start_date, end_date)
###
result, execution_time = measure_execution_time(classify_cities_with_most_job_offers, city_number_query, country_code, expertise_level, start_date, end_date)
execution_times.append(execution_time)
print("********************************************************")
print("Tiempo de ejecución - Requerimiento 4:", execution_time, "segundos")
print("********************************************************")


# Datos de ejemplo (tiempos de ejecución de dos requerimientos)
request = ['Requerimiento 1', 'Requerimiento 2', 'Requerimiento 3', 'Requerimiento 4']
plt.bar(request, execution_times, color=['blue', 'green', 'red', 'orange'])
plt.xlabel('Requerimientos')
plt.ylabel('Tiempo de Ejecución (segundos)')
plt.title('Comparación de Tiempos de Ejecución entre Requerimientos')
plt.show()