import time
import streamlit as st
import pandas as pd
import os
from matplotlib.font_manager import FontProperties
import plotly.express as px
from openai import OpenAI
from dateutil.relativedelta import relativedelta
from streamlit_extras.stylable_container import stylable_container

### Supplementary functions
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.20)

### Supplementary UI settings
st.set_page_config(layout="wide")
legend_title_font = FontProperties(size=14, weight='bold')
# Custom CSS to style the AskAI tab with light orange color
st.markdown("""
            <style>
                /* Style for the AskAI tab with rounded corners */
                .stTabs [data-baseweb="tab-list"] button[role="tab"]:first-child {
                background-color: #FFA07A; /* Light orange/salmon background */
                color: #333333; /* Darker text for better contrast */
                border-radius: 8px 8px 0 0; /* Rounded top corners */
                margin-bottom: -2px; /* Helps connect with content area */
                border: none;
                padding: 8px 16px; /* Add some padding for better appearance */
                }
            </style>
""",unsafe_allow_html=True)

### Data & API Inputs

## Access key

## Data
@st.cache_data
def load_data():
    # Simulating a time-consuming data loading process
    df = pd.read_excel(r'Combined Data - 3 years.xlsx')
    df['MonthYear'] = pd.to_datetime("01"+"-"+df['MonthShortName']+"-"+df['Year2'].astype(str),format='%d-%b-%Y').dt.date
    df['MonthYearShortForm'] = df['MonthShortName']+"-"+df['Year2'].astype(str)
    return df

## Call the function to load data (it will be cached, except for the first time)
combineddata = load_data()

combineddata['Country']='UAE'
## required variables derived from data
retailer = list(set(combineddata['Customer Name']))
allretailer = retailer.copy()
allretailer.insert(0,'All')

country = list(set(combineddata['Country']))
allcountry = country.copy()
allcountry.insert(0,'All')

month = list(set(combineddata['MonthShortName']))
reqmonthorder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
sortedmonths = sorted(month, key=lambda m: reqmonthorder.index(m))

## Subcategory - required variables derived from data
subcategory = list(set(combineddata['Subcategory']))
allsubcategory = subcategory.copy()
allsubcategory.insert(0,'All')

subcatretailer = list(set(combineddata['Customer Name']))
allsubcatretailer = subcatretailer.copy()
allsubcatretailer.insert(0,'All')

subcatcountry = list(set(combineddata['Country']))
allsubcatcountry = subcatcountry.copy()
allsubcatcountry.insert(0,'All')

subcatmonth = list(set(combineddata['MonthShortName']))
subcatreqmonthorder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
subcatsortedmonths = sorted(subcatmonth, key=lambda m: subcatreqmonthorder.index(m))

## Brand Performance
brand = list(set(combineddata['Brand Name']))
allbrand = brand.copy()
allbrand.insert(0,'All')

brandretailer = list(set(combineddata['Customer Name']))
allbrandretailer = brandretailer.copy()
allbrandretailer.insert(0,'All')

## Subcategory - required variables derived from data
brandsubcategory = list(set(combineddata['Subcategory']))
allbrandsubcategory = brandsubcategory.copy()
allbrandsubcategory.insert(0,'All')

brandmonth = list(set(combineddata['MonthShortName']))
brandreqmonthorder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
brandsortedmonths = sorted(brandmonth, key=lambda m: brandreqmonthorder.index(m))

brandoutlet = list(set(combineddata['Branch Name']))
allbrandoutlet = brandoutlet.copy()
allbrandoutlet.insert(0,'All')

### Backend support functions

### Cateogry dashboard

## Volume
def volume(reqdataset,retailerlist,countrylist,modelist,monthlist,retailer,country):
    
    ## Processing inputs
    if (len(retailerlist)==1) & (retailerlist[0]=="All"): retailerlist = retailer
    if (len(countrylist)==1) & (countrylist[0]=="All"): countrylist = country
    ytdmonthlistsubset = sortedmonths[0:sortedmonths.index(monthlist)+1]
    
    ## Filtering basis category and retailer
    requireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(retailerlist)) & (reqdataset['Country'].isin(countrylist))]
    
    ## Creating the required data basis inputs
    if modelist == 'YTD':
        
        ## Year graph
        vol = pd.pivot_table(requireddataset,values ='Sales Qty', index =['MonthYear'], columns = None, aggfunc = 'sum')
        # Filtering only for the required - creating month and year columns
        vol.index = pd.to_datetime(vol.index)
        vol['month'] = vol.index.strftime('%b') 
        vol['year'] = vol.index.strftime('%Y') 
        # Finally filering the required months
        vol = vol[vol['month'].isin(ytdmonthlistsubset)]
        
        # pivoting again to obtain yearly numbers
        yearvol = pd.pivot_table(vol,values = 'Sales Qty',index=['year'],columns = None, aggfunc = 'sum')
        yearvol = yearvol.reset_index()
        yearvol = yearvol.rename(columns={'year':'Years'})
        yearvol['Years'] = yearvol['Years'].astype(str)
        
        ## Month graph (only latest year)
        monthvol = vol.loc[vol['year']==max(vol['year']),['Sales Qty']]
        monthvol = monthvol.reset_index()
        monthvol = monthvol.rename(columns={'MonthYear':'Months'})
    
    else:
        # MAT Logic
        listoftuples = {}
        for year in set(combineddata['Year2']):
            startmonth = pd.to_datetime("01"+"-"+monthlist+"-"+str(year)).date() - pd.DateOffset(months=11)
            reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
            listoftuples[year] = reqmonthlist
        
        ## Year graph
        vol = pd.pivot_table(requireddataset,values ='Sales Qty', index =['MonthYear'], columns = None, aggfunc = 'sum')
        # Filtering only for the required rows
        vol = vol.reset_index()
        for t in listoftuples.keys():
            vol.loc[vol['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)
        
        # Cleaning up the dataframe where the MATYear is not present 12 times and rows where there is NA
        vol = vol.dropna(subset=['MATYear'])
        vol = vol.groupby('MATYear').filter(lambda x: len(x) >= 12)

        # pivoting again to obtain yearly numbers
        yearvol = pd.pivot_table(vol,values = 'Sales Qty',index=['MATYear'],columns = None, aggfunc = 'sum')
        yearvol = yearvol.reset_index()
        yearvol = yearvol.rename(columns={'MATYear':'Years'})
        yearvol['Years'] = yearvol['Years'].astype(str)
        
        ## Month graph (only latest year)
        monthvol = vol.loc[vol['MATYear']==max(vol['MATYear']),['MonthYear','Sales Qty']]
        monthvol = monthvol.reset_index()
        monthvol = monthvol.rename(columns={'MonthYear':'Months'})
        
    return([yearvol,monthvol])

## Value
def value(reqdataset,retailerlist,countrylist,modelist,monthlist,retailer,country):
    
    ## Processing inputs
    if (len(retailerlist)==1) & (retailerlist[0]=="All"): retailerlist = retailer
    if (len(countrylist)==1) & (countrylist[0]=="All"): countrylist = country
    ytdmonthlistsubset = sortedmonths[0:sortedmonths.index(monthlist)+1]
            
    ## Filtering basis category and retailer
    requireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(retailerlist)) & (reqdataset['Country'].isin(countrylist))]
    
    ## Creating the required data basis inputs
    if modelist == 'YTD':
        
        ## Year graph
        val = pd.pivot_table(requireddataset,values ='Sales AED', index =['MonthYear'], columns = None, aggfunc = 'sum')
        # Filtering only for the required - creating month and year columns
        val.index = pd.to_datetime(val.index)
        val['month'] = val.index.strftime('%b') 
        val['year'] = val.index.strftime('%Y') 
        # Finally filering the required months
        val = val[val['month'].isin(ytdmonthlistsubset)]
        
        # pivoting again to obtain yearly numbers
        yearval = pd.pivot_table(val,values = 'Sales AED',index=['year'],columns = None, aggfunc = 'sum')
        yearval = yearval.reset_index()
        yearval = yearval.rename(columns={'year':'Years'})
        yearval['Years'] = yearval['Years'].astype(str)
        
        ## Month graph (only latest year)
        monthval = val.loc[val['year']==max(val['year']),['Sales AED']]
        monthval = monthval.reset_index()
        monthval = monthval.rename(columns={'MonthYear':'Months'})
    
    else:
        # MAT Logic
        listoftuples = {}
        for year in set(combineddata['Year2']):
            startmonth = pd.to_datetime("01"+"-"+monthlist+"-"+str(year)).date() - pd.DateOffset(months=11)
            reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
            listoftuples[year] = reqmonthlist
        
        ## Year graph
        val = pd.pivot_table(requireddataset,values ='Sales AED', index =['MonthYear'], columns = None, aggfunc = 'sum')
        # Filtering only for the required rows
        val = val.reset_index()
        for t in listoftuples.keys():
            val.loc[val['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)
        
        # Cleaning up the dataframe where the MATYear is not present 12 times and rows where there is NA
        val = val.dropna(subset=['MATYear'])
        val = val.groupby('MATYear').filter(lambda x: len(x) >= 12)

        # pivoting again to obtain yearly numbers
        yearval = pd.pivot_table(val,values = 'Sales AED',index=['MATYear'],columns = None, aggfunc = 'sum')
        yearval = yearval.reset_index()
        yearval = yearval.rename(columns={'MATYear':'Years'})
        yearval['Years'] = yearval['Years'].astype(str)
        
        ## Month graph (only latest year)
        monthval = val.loc[val['MATYear']==max(val['MATYear']),['MonthYear','Sales AED']]
        monthval = monthval.reset_index()
        monthval = monthval.rename(columns={'MonthYear':'Months'})

    return([yearval,monthval])

## AI analysis
def categoryaianalysis(reqdfyear,reqdfmonth,val_vol,retailerlist,countrylist,modelist,monthlist):
    
    ## Pre-processing required inputs    
    reqdfyear_string = reqdfyear.to_string(index=False)
    reqdfmonth_string = reqdfmonth.to_string(index=False)

    retailerstring = ', '.join(retailerlist)
    countrystring = ', '.join(countrylist)
    modestring  = modelist
    monthstring = monthlist
    
    ## Creating category prompts
    if (retailerstring=='All') & (countrystring=='All'):
        detailedprompt = f"""Here are two tables where the first shows {val_vol} across years and the second across months for the latest years. The mode of analysis selected here is {modestring} and the same has been selected for {monthstring}"""
    else:
        detailedprompt = f"""Here are two tables where the first shows {val_vol} across years and the second across months for the latest years only for {retailerstring} and {countrystring}. The mode of analysis selected here is {modestring} and the same has been selected for {monthstring}"""
        
    ## Calling open AI function
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""
                    {detailedprompt}
                {reqdfyear_string}
                {reqdfmonth_string}
                
                With all the inputs, Can you provide a point-by-point commentary on the trends and key observations from this data? Derive insights from all the data. Strictly, not to exceed 3 points. These three should be short max of 1.5 lines. Don't start with "certainly!" or something related to it as this will be viewed by a user and not a part of chat'
                """
            }
        ]
    )
    return(completion.choices[0].message.content)


### Sub Category Dashboard

## Sub Category Volume
def subcategoryvolume(reqdataset,subcatretailerlist,subcatcountrylist,subcategorylist,subcatmodelist,subcatmonthlist,subcatretailer,subcatcountry,subcategory):
    
    ## Processing inputs
    if (len(subcatretailerlist)==1) & (subcatretailerlist[0]=="All"): subcatretailerlist = subcatretailer
    if (len(subcatcountrylist)==1) & (subcatcountrylist[0]=="All"): subcatcountrylist = subcatcountry
    if (len(subcategorylist)==1) & (subcategorylist[0]=="All"): subcategorylist = subcategory
    ytdmonthlistsubset = subcatsortedmonths[0:subcatsortedmonths.index(subcatmonthlist)+1]
    
    ## Filtering basis category and retailer
    subcatrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(subcatretailerlist)) & (reqdataset['Country'].isin(subcatcountrylist)) & (reqdataset['Subcategory'].isin(subcategorylist))]
    
    ## Creating the required data basis inputs
    if subcatmodelist == 'YTD':
        
        ## Year graph
        vol = pd.pivot_table(subcatrequireddataset,values ='Sales Qty', index =['MonthYear'], columns = ['Subcategory'], aggfunc = 'sum')
        # Filtering only for the required - creating month and year columns
        vol.index = pd.to_datetime(vol.index)
        vol['month'] = vol.index.strftime('%b') 
        vol['year'] = vol.index.strftime('%Y') 
        # Finally filering the required months
        vol = vol[vol['month'].isin(ytdmonthlistsubset)]
        
        # pivoting again to obtain yearly numbers
        subcatyearvol = vol.groupby('year').sum(numeric_only=True)
        subcatyearvol = subcatyearvol.reset_index()
        subcatyearvol = subcatyearvol.rename(columns={'year':'Years'})
        subcatyearvol['Years'] = subcatyearvol['Years'].astype(str)
        
        ## Month graph (only latest year)
        subcatmonthvol = vol.loc[vol['year']==max(vol['year'])].drop(['month', 'year'], axis=1)
        subcatmonthvol = subcatmonthvol.reset_index()
        subcatmonthvol = subcatmonthvol.rename(columns={'MonthYear':'Months'})
    
    else:
        # MAT Logic
        listoftuples = {}
        for year in set(combineddata['Year2']):
            startmonth = pd.to_datetime("01"+"-"+subcatmonthlist+"-"+str(year)).date() - pd.DateOffset(months=11)
            reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
            listoftuples[year] = reqmonthlist
        
        ## Year graph
        vol = pd.pivot_table(subcatrequireddataset,values ='Sales Qty', index =['MonthYear'], columns = ['Subcategory'], aggfunc = 'sum')
        # Filtering only for the required rows
        vol = vol.reset_index()
        for t in listoftuples.keys():
            vol.loc[vol['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)
        
        # Cleaning up the dataframe where the MATYear is not present 12 times and rows where there is NA
        vol = vol.dropna(subset=['MATYear'])
        vol = vol.groupby('MATYear').filter(lambda x: len(x) >= 12)

        # pivoting again to obtain yearly numbers
        subcatyearvol = vol.groupby('MATYear').sum(numeric_only=True)
        subcatyearvol = subcatyearvol.reset_index()
        subcatyearvol = subcatyearvol.rename(columns={'MATYear':'Years'})
        subcatyearvol['Years'] = subcatyearvol['Years'].astype(str)
        
        ## Month graph (only latest year)
        subcatmonthvol = vol.loc[vol['MATYear']==max(vol['MATYear'])].drop(['MATYear'],axis=1)
        subcatmonthvol = subcatmonthvol.reset_index()
        subcatmonthvol = subcatmonthvol.rename(columns={'MonthYear':'Months'})
        del subcatmonthvol['index']
    
    return([subcatyearvol,subcatmonthvol])

## Subcategory Value
def subcategoryvalue(reqdataset,subcatretailerlist,subcatcountrylist,subcategorylist,subcatmodelist,subcatmonthlist,subcatretailer,subcatcountry,subcategory):
    
    ## Processing inputs
    if (len(subcatretailerlist)==1) & (subcatretailerlist[0]=="All"): subcatretailerlist = subcatretailer
    if (len(subcatcountrylist)==1) & (subcatcountrylist[0]=="All"): subcatcountrylist = subcatcountry
    if (len(subcategorylist)==1) & (subcategorylist[0]=="All"): subcategorylist = subcategory
    ytdmonthlistsubset = subcatsortedmonths[0:subcatsortedmonths.index(subcatmonthlist)+1]
    
    ## Filtering basis category and retailer
    subcatrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(subcatretailerlist)) & (reqdataset['Country'].isin(subcatcountrylist)) & (reqdataset['Subcategory'].isin(subcategorylist))]
    
    ## Creating the required data basis inputs
    if subcatmodelist == 'YTD':
        
        ## Year graph
        val = pd.pivot_table(subcatrequireddataset,values ='Sales AED', index =['MonthYear'], columns = ['Subcategory'], aggfunc = 'sum')
        # Filtering only for the required - creating month and year columns
        val.index = pd.to_datetime(val.index)
        val['month'] = val.index.strftime('%b') 
        val['year'] = val.index.strftime('%Y') 
        # Finally filering the required months
        val = val[val['month'].isin(ytdmonthlistsubset)]
        
        # pivoting again to obtain yearly numbers
        subcatyearval = val.groupby('year').sum(numeric_only=True)

        subcatyearval = subcatyearval.reset_index()
        subcatyearval = subcatyearval.rename(columns={'year':'Years'})
        subcatyearval['Years'] = subcatyearval['Years'].astype(str)
        
        ## Month graph (only latest year)
        submonthval = val.loc[val['year']==max(val['year'])].drop(['month', 'year'], axis=1)
        submonthval = submonthval.reset_index()
        submonthval = submonthval.rename(columns={'MonthYear':'Months'})
    
    else:
        # MAT Logic
        listoftuples = {}
        for year in set(combineddata['Year2']):
            startmonth = pd.to_datetime("01"+"-"+subcatmonthlist+"-"+str(year)).date() - pd.DateOffset(months=11)
            reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
            listoftuples[year] = reqmonthlist
        
        ## Year graph
        val = pd.pivot_table(subcatrequireddataset,values ='Sales AED', index =['MonthYear'], columns = ['Subcategory'], aggfunc = 'sum')
        # Filtering only for the required rows
        val = val.reset_index()
        for t in listoftuples.keys():
            val.loc[val['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)
        
        # Cleaning up the dataframe where the MATYear is not present 12 times and rows where there is NA
        val = val.dropna(subset=['MATYear'])
        val = val.groupby('MATYear').filter(lambda x: len(x) >= 12)

        # pivoting again to obtain yearly numbers
        subcatyearval = val.groupby('MATYear').sum(numeric_only=True)
        subcatyearval = subcatyearval.reset_index()
        subcatyearval = subcatyearval.rename(columns={'MATYear':'Years'})
        subcatyearval['Years'] = subcatyearval['Years'].astype(str)
        
        ## Month graph (only latest year)
        submonthval = val.loc[val['MATYear']==max(val['MATYear'])].drop(['MATYear'],axis=1)
        submonthval = submonthval.reset_index()
        submonthval = submonthval.rename(columns={'MonthYear':'Months'})
        del submonthval['index']
        
    return([subcatyearval,submonthval])

## AI Analysis
def subcategoryaianalysis(reqdfyear,reqdfmonth,val_vol,retailerlist,countrylist,subcategorylist,modelist,monthlist):
    
    ## Pre-processing required inputs    
    reqdfyear_string = reqdfyear.to_string(index=False)
    reqdfmonth_string = reqdfmonth.to_string(index=False)

    retailerstring = ', '.join(retailerlist)
    countrystring = ', '.join(countrylist)
    subcategorystring = ', '.join(subcategorylist)
    modestring  = modelist
    monthstring = monthlist
    
    ## Creating category prompts
    if (retailerstring=='All') & (countrystring=='All') & (subcategorystring=='All'):
        detailedprompt = f"""Here are two tables where the first shows {val_vol} across years and the second across months for the latest years. The mode of analysis selected here is {modestring} and the same has been selected for {monthstring}"""
    else:
        detailedprompt = f"""Here are two tables where the first shows {val_vol} across years and the second across months for the latest years only for {retailerstring} retailers, {countrystring} countries and {subcategorystring} sub categories. The mode of analysis selected here is {modestring} and the same has been selected for {monthstring}"""
    
    ## Calling open AI function
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""
                    {detailedprompt}
                {reqdfyear_string}
                {reqdfmonth_string}
                
                With all the inputs, Can you provide a point-by-point commentary on the trends and key observations from this data? Derive insights from all the data. Strictly, not to exceed 3 points. These three should be short max of 1.5 lines. Don't start with "certainly!" or something related to it as this will be viewed by a user and not a part of chat'
                """
            }
        ]
    )
    return(completion.choices[0].message.content)

### Brand Performance backend

## Brand chart
def brandpievol(reqdataset, brandlist, brandretailerlist, brandsubcategorylist, brandoutletlist, brandregionlist, brandmodelist, brandmonthlist):
    
    ## Processing inputs
    if (len(brandlist)==1) & (brandlist[0]=="All"): brandlistv2 = brand
    else: brandlistv2 = brandlist
    if (len(brandretailerlist)==1) & (brandretailerlist[0]=="All"): brandretailerlist = brandretailer
    if (len(brandsubcategorylist)==1) & (brandsubcategorylist[0]=="All"): brandsubcategorylist = brandsubcategory
    if (len(brandoutletlist)==1) & (brandoutletlist[0]=="All"): brandoutletlist = brandoutlet
    ytdmonthlistsubset =  brandsortedmonths[0:brandsortedmonths.index(brandmonthlist)+1]
    threemonthlist = pd.to_datetime("01"+"-"+brandmonthlist+"-"+str(max(reqdataset['Year2'])),format='%d-%b-%Y')
    
    ## Filtering basis category and retailer
    brandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Brand Name'].isin(brandlistv2)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
    
    ## Creating the required data basis inputs for MAT/YTD
    if brandmodelist == 'YTD':
        
        brand_totals = brandrequireddataset[(brandrequireddataset['MonthShortName'].isin(ytdmonthlistsubset)) & (brandrequireddataset['Year2']==max(brandrequireddataset['Year2']))].groupby('Brand Name')['Sales AED'].sum().reset_index()
        if (len(brandlist)==1) & (brandlist[0]=="All"):
            brand_totals['Share'] = brand_totals['Sales AED'] / brand_totals['Sales AED'].sum()
            # Only top 7
            brand_totals = brand_totals.sort_values(by='Share', ascending=False).reset_index(drop=True)
            top7 = brand_totals.head(7).copy()
            others_sum = brand_totals.iloc[7:]['Share'].sum()
            # Add "Others" row
            top7.loc[top7.shape[0]+1,'Brand Name'] = 'Others'
            top7.loc[top7.shape[0],'Share'] = others_sum
            # Final DataFrame
            pieresult = top7[['Brand Name', 'Share']]
        else:
            alt_brandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
            alt_brand_totals = alt_brandrequireddataset[(alt_brandrequireddataset['MonthShortName'].isin(ytdmonthlistsubset)) & (alt_brandrequireddataset['Year2']==max(alt_brandrequireddataset['Year2']))].groupby('Brand Name')['Sales AED'].sum().reset_index()
            alt_brand_totals['Share'] = alt_brand_totals['Sales AED'] / alt_brand_totals['Sales AED'].sum()
            # Grouping brands other than selected brands to others
            alt_brand_totals.loc[alt_brand_totals.shape[0]+1,'Brand Name'] = 'Others'
            alt_brand_totals.loc[alt_brand_totals.shape[0],'Share'] = alt_brand_totals.loc[~alt_brand_totals['Brand Name'].isin(brandlistv2),'Share'].sum()
            pieresult = alt_brand_totals[['Brand Name', 'Share']]
            brandlistv2_change = brandlistv2.copy()
            brandlistv2_change.append('Others')
            pieresult = pieresult.loc[pieresult['Brand Name'].isin(brandlistv2_change)]

    else:
        
        # MAT Logic
        listoftuples = {}
        for year in set(combineddata['Year2']):
            startmonth = pd.to_datetime("01"+"-"+brandmonthlist+"-"+str(year)).date() - pd.DateOffset(months=11)
            reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
            listoftuples[year] = reqmonthlist
        
        val = pd.pivot_table(brandrequireddataset,values ='Sales AED', index =['MonthYear'], columns = ['Brand Name'], aggfunc = 'sum')

        # Filtering only for the required rows
        val = val.reset_index()
        for t in listoftuples.keys():
           val.loc[val['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)

        # Cleaning up the dataframe where the MATYear is not present 12 times and rows where there is NA
        val = val.dropna(subset=['MATYear']).groupby('MATYear').filter(lambda x: len(x) >= 12)
        # totalling at a brand level
        brand_totals = val.loc[val['MATYear']==max(val['MATYear'])].drop(['MATYear','MonthYear'],axis=1)
        brand_totals_mat = brand_totals.sum().to_frame().reset_index().rename(columns={0:'Sales AED'})
                    
        brand_totals_mat['Share'] = brand_totals_mat['Sales AED'] / brand_totals_mat['Sales AED'].sum()
        # Only top 7
        brand_totals_mat = brand_totals_mat.sort_values(by='Share', ascending=False).reset_index(drop=True)
        top7 = brand_totals_mat.head(7).copy()
        others_sum = brand_totals_mat.iloc[7:]['Share'].sum()
        # Add "Others" row
        top7.loc[top7.shape[0]+1,'Brand Name'] = 'Others'
        top7.loc[top7.shape[0],'Share'] = others_sum
        # Final DataFrame
        pieresult = top7[['Brand Name', 'Share']]
        
        if (len(brandlist)==1) & (brandlist[0]=="All"):
            b=1
            b=b+1
        else:
            alt_brandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
            alt_val = pd.pivot_table(alt_brandrequireddataset,values ='Sales AED', index =['MonthYear'], columns = ['Brand Name'], aggfunc = 'sum')

            # Filtering only for the required rows
            alt_val = alt_val.reset_index()
            for t in listoftuples.keys():
               alt_val.loc[val['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)

            # Cleaning up the dataframe where the MATYear is not present 12 times and rows where there is NA
            alt_val = alt_val.dropna(subset=['MATYear']).groupby('MATYear').filter(lambda x: len(x) >= 12)
            # totalling at a brand level
            alt_brand_totals = alt_val.loc[alt_val['MATYear']==max(alt_val['MATYear'])].drop(['MATYear','MonthYear'],axis=1)
            alt_brand_totals_mat = alt_brand_totals.sum().to_frame().reset_index().rename(columns={0:'Sales AED'})
            alt_brand_totals_mat['Share'] = alt_brand_totals_mat['Sales AED'] / alt_brand_totals_mat['Sales AED'].sum()
            # Grouping brands other than selected brands to others
            alt_brand_totals_mat.loc[alt_brand_totals_mat.shape[0]+1,'Brand Name'] = 'Others'
            alt_brand_totals_mat.loc[alt_brand_totals_mat.shape[0],'Share'] = alt_brand_totals_mat.loc[~alt_brand_totals_mat['Brand Name'].isin(brandlistv2),'Share'].sum()
            pieresult = alt_brand_totals_mat[['Brand Name', 'Share']]
            brandlistv2_change = brandlistv2.copy()
            brandlistv2_change.append('Others')
            pieresult = pieresult.loc[pieresult['Brand Name'].isin(brandlistv2_change)]

    ## Creating the required data basis inputs for 3M
    threemonthreqlist = []
    for i in range(0,3):
        threemonthreqlist.append((threemonthlist - relativedelta(months=i)).strftime('%b-%Y'))
    
    brand_totals_3M = brandrequireddataset[(brandrequireddataset['MonthYearShortForm'].isin(threemonthreqlist))].groupby('Brand Name')['Sales AED'].sum().reset_index()
    if (len(brandlist)==1) & (brandlist[0]=="All"):

        brand_totals_3M['Share'] = brand_totals_3M['Sales AED'] / brand_totals_3M['Sales AED'].sum()
        # Only top 7
        brand_totals_3M = brand_totals_3M.sort_values(by='Share', ascending=False).reset_index(drop=True)
        top7_3M = brand_totals_3M.head(7).copy()
        others_sum_3M = brand_totals_3M.iloc[7:]['Share'].sum()
        # Add "Others" row
        top7_3M.loc[top7_3M.shape[0]+1,'Brand Name'] = 'Others'
        top7_3M.loc[top7_3M.shape[0],'Share'] = others_sum_3M
        # Final DataFrame
        pieresult_3M = top7_3M[['Brand Name', 'Share']]
    else:
        
        alt_brandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
        alt_brand_totals_3M = alt_brandrequireddataset[(alt_brandrequireddataset['MonthYearShortForm'].isin(threemonthreqlist))].groupby('Brand Name')['Sales AED'].sum().reset_index()
        alt_brand_totals_3M['Share'] = alt_brand_totals_3M['Sales AED'] / alt_brand_totals_3M['Sales AED'].sum()
        
        # Grouping brands other than selected brands to others
        alt_brand_totals_3M.loc[alt_brand_totals_3M.shape[0]+1,'Brand Name'] = 'Others'
        alt_brand_totals_3M.loc[alt_brand_totals_3M.shape[0],'Share'] = alt_brand_totals_3M.loc[~alt_brand_totals_3M['Brand Name'].isin(brandlistv2),'Share'].sum()
        pieresult_3M = alt_brand_totals_3M[['Brand Name', 'Share']]
        brandlistv2_change = brandlistv2.copy()
        brandlistv2_change.append('Others')
        pieresult_3M = pieresult_3M.loc[pieresult_3M['Brand Name'].isin(brandlistv2_change)]

    ## Creating the required data basis inputs for 1M/current month        
    brand_totals_1M = brandrequireddataset[(brandrequireddataset['MonthYearShortForm']==brandmonthlist+'-'+str(max(brandrequireddataset['Year2'])))].groupby('Brand Name')['Sales AED'].sum().reset_index()
    if (len(brandlist)==1) & (brandlist[0]=="All"):

        brand_totals_1M['Share'] = brand_totals_1M['Sales AED'] / brand_totals_1M['Sales AED'].sum()
        # Only top 7
        brand_totals_1M = brand_totals_1M.sort_values(by='Share', ascending=False).reset_index(drop=True)
        top7_1M = brand_totals_1M.head(7).copy()
        others_sum_1M = brand_totals_1M.iloc[7:]['Share'].sum()
        # Add "Others" row
        top7_1M.loc[top7_1M.shape[0]+1,'Brand Name'] = 'Others'
        top7_1M.loc[top7_1M.shape[0],'Share'] = others_sum_1M
        # Final DataFrame
        pieresult_1M = top7_1M[['Brand Name', 'Share']]
    
    else:
        alt_brandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
        alt_brand_totals_1M = alt_brandrequireddataset[(alt_brandrequireddataset['MonthYearShortForm']==brandmonthlist+'-'+str(max(alt_brandrequireddataset['Year2'])))].groupby('Brand Name')['Sales AED'].sum().reset_index()

        alt_brand_totals_1M['Share'] = alt_brand_totals_1M['Sales AED'] / alt_brand_totals_1M['Sales AED'].sum()
        
        # Grouping brands other than selected brands to others
        alt_brand_totals_1M.loc[alt_brand_totals_1M.shape[0]+1,'Brand Name'] = 'Others'
        alt_brand_totals_1M.loc[alt_brand_totals_1M.shape[0],'Share'] = alt_brand_totals_1M.loc[~alt_brand_totals_1M['Brand Name'].isin(brandlistv2),'Share'].sum()
        pieresult_1M = alt_brand_totals_1M[['Brand Name', 'Share']]
        brandlistv2_change = brandlistv2.copy()
        brandlistv2_change.append('Others')
        pieresult_1M = pieresult_1M.loc[pieresult_1M['Brand Name'].isin(brandlistv2_change)]

    return([pieresult,pieresult_3M,pieresult_1M])

## Brand table
def brand_table_data_vol(reqdataset, brandlist, brandretailerlist, brandsubcategorylist, brandoutletlist, brandregionlist, brandmodelist, brandmonthlist):

    ## Processing inputs
    if (len(brandlist)==1) & (brandlist[0]=="All"): brandlistv2 = brand
    else: brandlistv2 = brandlist
    if (len(brandretailerlist)==1) & (brandretailerlist[0]=="All"): brandretailerlist = brandretailer
    if (len(brandsubcategorylist)==1) & (brandsubcategorylist[0]=="All"): brandsubcategorylist = brandsubcategory
    if (len(brandoutletlist)==1) & (brandoutletlist[0]=="All"): brandoutletlist = brandoutlet
    ytdmonthlistsubset =  brandsortedmonths[0:brandsortedmonths.index(brandmonthlist)+1]
    threemonthlist = pd.to_datetime("01"+"-"+brandmonthlist+"-"+str(max(reqdataset['Year2'])),format='%d-%b-%Y')
    onemonthlist = pd.to_datetime("01"+"-"+brandmonthlist+"-"+str(max(reqdataset['Year2'])),format='%d-%b-%Y')

    ## Filtering basis basic variables
    brandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Brand Name'].isin(brandlistv2)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
   
    if brandmodelist == 'YTD':
        brand_totals = brandrequireddataset[(brandrequireddataset['MonthShortName'].isin(ytdmonthlistsubset)) & (brandrequireddataset['Year2'].isin([max(brandrequireddataset['Year2']),brandrequireddataset['Year2'][brandrequireddataset['Year2'] < brandrequireddataset['Year2'].max()].max()]))].groupby(['Brand Name','Year2'])['Sales Qty'].sum().reset_index()
        brand_totals_pivot = pd.pivot_table(brand_totals,values ='Sales Qty', index =['Brand Name'], columns = ['Year2'], aggfunc = 'sum')
        cols = list(brand_totals_pivot.columns)
        cols[0] = 'Year Ago'
        cols[1] = 'This Year'
        brand_totals_pivot.columns = cols
        # % change
        brand_totals_pivot['% Change'] = (brand_totals_pivot['This Year'] - brand_totals_pivot['Year Ago']) / brand_totals_pivot['Year Ago']
        top10_brands_vol_ytdmat = brand_totals_pivot.sort_values(by='This Year',ascending=False).head(10)
        top10_brands_vol_ytdmat['% Change'] = top10_brands_vol_ytdmat['% Change'].apply(lambda x: f"{x:.2%}")
        
        # Changing Market Share%
        if (len(brandlist)==1) & (brandlist[0]=="All"):
            a = 1
            a = a+1
        else:
            altbrandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
            alt_brand_totals = altbrandrequireddataset[(altbrandrequireddataset['MonthShortName'].isin(ytdmonthlistsubset)) & (altbrandrequireddataset['Year2'].isin([max(altbrandrequireddataset['Year2']),altbrandrequireddataset['Year2'][altbrandrequireddataset['Year2'] < altbrandrequireddataset['Year2'].max()].max()]))].groupby(['Brand Name','Year2'])['Sales Qty'].sum().reset_index()
            alt_brand_totals_pivot = pd.pivot_table(alt_brand_totals,values ='Sales Qty', index =['Brand Name'], columns = ['Year2'], aggfunc = 'sum')
            cols = list(alt_brand_totals_pivot.columns)
            cols[0] = 'Year Ago'
            cols[1] = 'This Year'
            alt_brand_totals_pivot.columns = cols
        
        # % share
        if (len(brandlist)==1) & (brandlist[0]=="All"):
            brand_totals_pivot['% TY Share'] = brand_totals_pivot['This Year'] / brand_totals_pivot['This Year'].sum()
            brand_totals_pivot['% YA Share'] = brand_totals_pivot['Year Ago'] / brand_totals_pivot['Year Ago'].sum()
            brand_totals_pivot['% Change'] = (brand_totals_pivot['% TY Share'] - brand_totals_pivot['% YA Share'])/brand_totals_pivot['% YA Share']
            top10_brands_volshare_ytdmat = brand_totals_pivot.sort_values(by='This Year',ascending=False).head(10)[['% YA Share','% TY Share','% Change']]
            top10_brands_volshare_ytdmat['% Change'] = top10_brands_volshare_ytdmat['% Change'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% TY Share'] = top10_brands_volshare_ytdmat['% TY Share'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% YA Share'] = top10_brands_volshare_ytdmat['% YA Share'].apply(lambda x: f"{x:.2%}")
        else:
            brand_totals_pivot['% TY Share'] = brand_totals_pivot['This Year'] / alt_brand_totals_pivot['This Year'].sum()
            brand_totals_pivot['% YA Share'] = brand_totals_pivot['Year Ago'] / alt_brand_totals_pivot['Year Ago'].sum()
            brand_totals_pivot['% Change'] = (brand_totals_pivot['% TY Share'] - brand_totals_pivot['% YA Share'])/brand_totals_pivot['% YA Share']
            top10_brands_volshare_ytdmat = brand_totals_pivot.sort_values(by='This Year',ascending=False).head(10)[['% YA Share','% TY Share','% Change']]
            top10_brands_volshare_ytdmat['% Change'] = top10_brands_volshare_ytdmat['% Change'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% TY Share'] = top10_brands_volshare_ytdmat['% TY Share'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% YA Share'] = top10_brands_volshare_ytdmat['% YA Share'].apply(lambda x: f"{x:.2%}")
    else:
        # MAT Logic
        listoftuples = {}
        for year in set(combineddata['Year2']):
            startmonth = pd.to_datetime("01"+"-"+brandmonthlist+"-"+str(year)).date() - pd.DateOffset(months=11)
            reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
            listoftuples[year] = reqmonthlist
        val = pd.pivot_table(brandrequireddataset,values ='Sales Qty', index =['MonthYear'], columns = ['Brand Name'], aggfunc = 'sum')

        # Filtering only for the required rows
        val = val.reset_index()
        for t in listoftuples.keys():
           val.loc[val['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)
        val = val.dropna(subset=['MATYear']).groupby('MATYear').filter(lambda x: len(x) >= 12)
        
        # totalling at a brand level
        brand_totals = val.loc[val['MATYear'].isin([max(val['MATYear']),val['MATYear'][val['MATYear'] < val['MATYear'].max()].max()])].drop(['MonthYear'],axis=1)
        brand_totals_pivot = brand_totals.melt(id_vars='MATYear', var_name='Brand', value_name='Value').groupby(['Brand','MATYear'], as_index=False)['Value'].sum()
        brand_totals_pivot = pd.pivot_table(brand_totals_pivot,values='Value',index='Brand',columns='MATYear',aggfunc='sum')
        
        cols = list(brand_totals_pivot.columns)
        cols[0] = 'Year Ago'
        cols[1] = 'This Year'
        brand_totals_pivot.columns = cols
        
        brand_totals_pivot['% Change'] = (brand_totals_pivot['This Year'] - brand_totals_pivot['Year Ago']) / brand_totals_pivot['Year Ago']
        top10_brands_vol_ytdmat = brand_totals_pivot.sort_values(by='This Year',ascending=False).head(10)
        top10_brands_vol_ytdmat['% Change'] = top10_brands_vol_ytdmat['% Change'].apply(lambda x: f"{x:.2%}")
        
        # Changing Market Share%
        if (len(brandlist)==1) & (brandlist[0]=="All"):
            a = 1
            a = a+1
        else:
            altbrandrequireddataset = reqdataset.loc[(reqdataset['Customer Name'].isin(brandretailerlist)) & (reqdataset['Subcategory'].isin(brandsubcategorylist)) & (reqdataset['Branch Name'].isin(brandoutletlist))]
            alt_val = pd.pivot_table(altbrandrequireddataset,values ='Sales Qty', index =['MonthYear'], columns = ['Brand Name'], aggfunc = 'sum')

            # Filtering only for the required rows
            alt_val = alt_val.reset_index()
            for t in listoftuples.keys():
               alt_val.loc[alt_val['MonthYear'].astype(str).isin(listoftuples[t]),'MATYear'] = str(t)
            alt_val = alt_val.dropna(subset=['MATYear']).groupby('MATYear').filter(lambda x: len(x) >= 12)
            
            # totalling at a brand level
            alt_brand_totals = alt_val.loc[val['MATYear'].isin([max(alt_val['MATYear']),alt_val['MATYear'][alt_val['MATYear'] < alt_val['MATYear'].max()].max()])].drop(['MonthYear'],axis=1)
            alt_brand_totals_pivot = alt_brand_totals.melt(id_vars='MATYear', var_name='Brand', value_name='Value').groupby(['Brand','MATYear'], as_index=False)['Value'].sum()
            alt_brand_totals_pivot = pd.pivot_table(alt_brand_totals_pivot,values='Value',index='Brand',columns='MATYear',aggfunc='sum')
            
            cols = list(alt_brand_totals_pivot.columns)
            cols[0] = 'Year Ago'
            cols[1] = 'This Year'
            alt_brand_totals_pivot.columns = cols
        
        # % share
        if (len(brandlist)==1) & (brandlist[0]=="All"):

            brand_totals_pivot['% TY Share'] = brand_totals_pivot['This Year'] / brand_totals_pivot['This Year'].sum()
            brand_totals_pivot['% YA Share'] = brand_totals_pivot['Year Ago'] / brand_totals_pivot['Year Ago'].sum()
    
            brand_totals_pivot['% Change'] = (brand_totals_pivot['% TY Share'] - brand_totals_pivot['% YA Share'])/brand_totals_pivot['% YA Share']
            top10_brands_volshare_ytdmat = brand_totals_pivot.sort_values(by='This Year',ascending=False).head(10)[['% YA Share','% TY Share','% Change']]
            top10_brands_volshare_ytdmat['% Change'] = top10_brands_volshare_ytdmat['% Change'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% TY Share'] = top10_brands_volshare_ytdmat['% TY Share'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% YA Share'] = top10_brands_volshare_ytdmat['% YA Share'].apply(lambda x: f"{x:.2%}")
        
        else:
            
            brand_totals_pivot['% TY Share'] = brand_totals_pivot['This Year'] / alt_brand_totals_pivot['This Year'].sum()
            brand_totals_pivot['% YA Share'] = brand_totals_pivot['Year Ago'] / alt_brand_totals_pivot['Year Ago'].sum()
    
            brand_totals_pivot['% Change'] = (brand_totals_pivot['% TY Share'] - brand_totals_pivot['% YA Share'])/brand_totals_pivot['% YA Share']
            top10_brands_volshare_ytdmat = brand_totals_pivot.sort_values(by='This Year',ascending=False).head(10)[['% YA Share','% TY Share','% Change']]
            top10_brands_volshare_ytdmat['% Change'] = top10_brands_volshare_ytdmat['% Change'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% TY Share'] = top10_brands_volshare_ytdmat['% TY Share'].apply(lambda x: f"{x:.2%}")
            top10_brands_volshare_ytdmat['% YA Share'] = top10_brands_volshare_ytdmat['% YA Share'].apply(lambda x: f"{x:.2%}")

    ## 3M
    threemonthreqlist = []
    for i in range(0,6):
        threemonthreqlist.append((threemonthlist - relativedelta(months=i)).strftime('%b-%Y'))
    
    if (len(brandlist)==1) & (brandlist[0]=="All"):
        a+=1
    else:
        alt_brand_totals_3M = altbrandrequireddataset[(altbrandrequireddataset['MonthYearShortForm'].isin(threemonthreqlist))]        
        alt_brand_totals_3M.loc[alt_brand_totals_3M['MonthYearShortForm'].isin(threemonthreqlist[0:3]),'3M Indicator'] = 'Current 3M TY'
        alt_brand_totals_3M.loc[alt_brand_totals_3M['MonthYearShortForm'].isin(threemonthreqlist[3:6]),'3M Indicator'] = 'Previous 3M TY'
        alt_brand_totals_3M_pivot = alt_brand_totals_3M.groupby(['Brand Name','3M Indicator'])['Sales Qty'].sum().reset_index()
        alt_brand_totals_3M_pivot = pd.pivot_table(alt_brand_totals_3M_pivot,values ='Sales Qty', index =['Brand Name'], columns = ['3M Indicator'], aggfunc = 'sum')

    brand_totals_3M = brandrequireddataset[(brandrequireddataset['MonthYearShortForm'].isin(threemonthreqlist))]        
    brand_totals_3M.loc[brand_totals_3M['MonthYearShortForm'].isin(threemonthreqlist[0:3]),'3M Indicator'] = 'Current 3M TY'
    brand_totals_3M.loc[brand_totals_3M['MonthYearShortForm'].isin(threemonthreqlist[3:6]),'3M Indicator'] = 'Previous 3M TY'
    brand_totals_3M_pivot = brand_totals_3M.groupby(['Brand Name','3M Indicator'])['Sales Qty'].sum().reset_index()
    brand_totals_3M_pivot = pd.pivot_table(brand_totals_3M_pivot,values ='Sales Qty', index =['Brand Name'], columns = ['3M Indicator'], aggfunc = 'sum')
    
    # Market Share
    if (len(brandlist)==1) & (brandlist[0]=="All"):
        brand_totals_3M_pivot['% Current'] = brand_totals_3M_pivot['Current 3M TY'] / brand_totals_3M_pivot['Current 3M TY'].sum()
        brand_totals_3M_pivot['% Previous'] = brand_totals_3M_pivot['Previous 3M TY'] / brand_totals_3M_pivot['Previous 3M TY'].sum()
        brand_totals_3M_pivot['% Change'] = (brand_totals_3M_pivot['% Current'] - brand_totals_3M_pivot['% Previous'])/brand_totals_3M_pivot['% Previous']
        top10_brands_vol_3M = brand_totals_3M_pivot[brand_totals_3M_pivot.index.isin(list(top10_brands_volshare_ytdmat.index))].sort_values(by='Current 3M TY',ascending=False)[['% Current','% Previous','% Change']]
        top10_brands_vol_3M['% Change'] = top10_brands_vol_3M['% Change'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_3M['% Current'] = top10_brands_vol_3M['% Current'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_3M['% Previous'] = top10_brands_vol_3M['% Previous'].apply(lambda x: f"{x:.2%}")
    else:
        brand_totals_3M_pivot['% Current'] = brand_totals_3M_pivot['Current 3M TY'] / alt_brand_totals_3M_pivot['Current 3M TY'].sum()
        brand_totals_3M_pivot['% Previous'] = brand_totals_3M_pivot['Previous 3M TY'] / alt_brand_totals_3M_pivot['Previous 3M TY'].sum()
        brand_totals_3M_pivot['% Change'] = (brand_totals_3M_pivot['% Current'] - brand_totals_3M_pivot['% Previous'])/brand_totals_3M_pivot['% Previous']
        top10_brands_vol_3M = brand_totals_3M_pivot[brand_totals_3M_pivot.index.isin(list(top10_brands_volshare_ytdmat.index))].sort_values(by='Current 3M TY',ascending=False)[['% Current','% Previous','% Change']]
        top10_brands_vol_3M['% Change'] = top10_brands_vol_3M['% Change'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_3M['% Current'] = top10_brands_vol_3M['% Current'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_3M['% Previous'] = top10_brands_vol_3M['% Previous'].apply(lambda x: f"{x:.2%}")

    ## 1M
    onemonthreqlist = []
    for i in range(0,2):
        onemonthreqlist.append((onemonthlist - relativedelta(months=i)).strftime('%b-%Y'))
    if (len(brandlist)==1) & (brandlist[0]=="All"):
        a+=1
    else:
       alt_brand_totals_1M = altbrandrequireddataset[(altbrandrequireddataset['MonthYearShortForm'].isin(onemonthreqlist))]
       alt_brand_totals_1M.loc[alt_brand_totals_1M['MonthYearShortForm']==onemonthreqlist[0],'1M Indicator'] = 'Current Month'
       alt_brand_totals_1M.loc[alt_brand_totals_1M['MonthYearShortForm']==onemonthreqlist[1],'1M Indicator'] = 'Previous Month'
       alt_brand_totals_1M_pivot = alt_brand_totals_1M.groupby(['Brand Name','1M Indicator'])['Sales Qty'].sum().reset_index()
       alt_brand_totals_1M_pivot = pd.pivot_table(alt_brand_totals_1M,values ='Sales Qty', index =['Brand Name'], columns = ['1M Indicator'], aggfunc = 'sum')        
    
    brand_totals_1M = brandrequireddataset[(brandrequireddataset['MonthYearShortForm'].isin(onemonthreqlist))]
    brand_totals_1M.loc[brand_totals_1M['MonthYearShortForm']==onemonthreqlist[0],'1M Indicator'] = 'Current Month'
    brand_totals_1M.loc[brand_totals_1M['MonthYearShortForm']==onemonthreqlist[1],'1M Indicator'] = 'Previous Month'
    brand_totals_1M_pivot = brand_totals_1M.groupby(['Brand Name','1M Indicator'])['Sales Qty'].sum().reset_index()
    brand_totals_1M_pivot = pd.pivot_table(brand_totals_1M_pivot,values ='Sales Qty', index =['Brand Name'], columns = ['1M Indicator'], aggfunc = 'sum')
    # Market Share
    if (len(brandlist)==1) & (brandlist[0]=="All"):
        brand_totals_1M_pivot['% Current'] = brand_totals_1M_pivot['Current Month'] / brand_totals_1M_pivot['Current Month'].sum()
        brand_totals_1M_pivot['% Previous'] = brand_totals_1M_pivot['Previous Month'] / brand_totals_1M_pivot['Previous Month'].sum()
        brand_totals_1M_pivot['% Change'] = (brand_totals_1M_pivot['% Current'] - brand_totals_1M_pivot['% Previous'])/brand_totals_1M_pivot['% Previous']
        top10_brands_vol_1M = brand_totals_1M_pivot[brand_totals_1M_pivot.index.isin(list(top10_brands_volshare_ytdmat.index))].sort_values(by='Current Month',ascending=False)[['% Current','% Previous','% Change']]
        top10_brands_vol_1M['% Change'] = top10_brands_vol_1M['% Change'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_1M['% Current'] = top10_brands_vol_1M['% Current'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_1M['% Previous'] = top10_brands_vol_1M['% Previous'].apply(lambda x: f"{x:.2%}")
    else:
        brand_totals_1M_pivot['% Current'] = brand_totals_1M_pivot['Current Month'] / alt_brand_totals_1M_pivot['Current Month'].sum()
        brand_totals_1M_pivot['% Previous'] = brand_totals_1M_pivot['Previous Month'] / alt_brand_totals_1M_pivot['Previous Month'].sum()
        brand_totals_1M_pivot['% Change'] = (brand_totals_1M_pivot['% Current'] - brand_totals_1M_pivot['% Previous'])/brand_totals_1M_pivot['% Previous']
        top10_brands_vol_1M = brand_totals_1M_pivot[brand_totals_1M_pivot.index.isin(list(top10_brands_volshare_ytdmat.index))].sort_values(by='Current Month',ascending=False)[['% Current','% Previous','% Change']]
        top10_brands_vol_1M['% Change'] = top10_brands_vol_1M['% Change'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_1M['% Current'] = top10_brands_vol_1M['% Current'].apply(lambda x: f"{x:.2%}")
        top10_brands_vol_1M['% Previous'] = top10_brands_vol_1M['% Previous'].apply(lambda x: f"{x:.2%}")
    
    return(top10_brands_vol_ytdmat,top10_brands_volshare_ytdmat,top10_brands_vol_3M,top10_brands_vol_1M)
  
### KPI dashboard backend
def kpis(combineddata):
    
    # Identify recent month and year from combineddata
    month=combineddata.loc[combineddata['MonthYear']==combineddata['MonthYear'].max()].iloc[0]['MonthShortName']
    year=combineddata.loc[combineddata['MonthYear']==combineddata['MonthYear'].max()].iloc[0]['Year2']
    
    # MAT Logic
    listoftuples = {}
    for year_val in [year,year-1]:
        startmonth = pd.to_datetime("01"+"-"+month+"-"+str(year_val)).date() - pd.DateOffset(months=11)
        reqmonthlist = tuple(pd.date_range(start=startmonth, periods=12, freq='MS').strftime('%Y-%m-%d'))
        listoftuples[year_val] = reqmonthlist
         
    top5retailernames = list(combineddata[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))].groupby('Customer Name')['Sales Qty'].sum().reset_index().sort_values(by='Sales Qty', ascending=False)['Customer Name'])[0:4]
    top2brandnames = list(combineddata[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))].groupby('Brand Name')['Sales Qty'].sum().reset_index().sort_values(by='Sales Qty', ascending=False)['Brand Name'])[0:2]

    # overall sales
    latestoverallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])),'Sales Qty'])
    previousoverallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])),'Sales Qty'])
    percentchangeinsales = (latestoverallsales - previousoverallsales)/previousoverallsales
    
    # brand sales
    latestbrandsales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON'),'Sales Qty'])
    previousbrandsales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']=='LAMB WESTON'),'Sales Qty'])
    percentchangeinsalesbrand = (latestbrandsales - previousbrandsales)/previousbrandsales
    
    # retailer 1 overall sales
    latestretailer1overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])) & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    previousretailer1overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])) & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    percentchangeinsalesretailer1overall = (latestretailer1overallsales - previousretailer1overallsales)/previousretailer1overallsales
    
    # retailer 1 sales
    latestretailer1sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    previousretailer1sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    percentchangeinsalesretailer1 = (latestretailer1sales - previousretailer1sales)/previousretailer1sales
    
    # retailer 2 overall sales
    latestretailer2overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])) & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    previousretailer2overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])) & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    percentchangeinsalesretailer2overall = (latestretailer2overallsales - previousretailer2overallsales)/previousretailer2overallsales
    
    # retailer 2 sales
    latestretailer2sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    previousretailer2sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    percentchangeinsalesretailer2 = (latestretailer2sales - previousretailer2sales)/previousretailer2sales
    
    # retailer 3 overall sales
    latestretailer3overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])) & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    previousretailer3overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])) & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    percentchangeinsalesretailer3overall = (latestretailer3overallsales - previousretailer3overallsales)/previousretailer3overallsales
    
    # retailer 3 sales
    latestretailer3sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    previousretailer3sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    percentchangeinsalesretailer3 = (latestretailer3sales - previousretailer3sales)/previousretailer3sales
    
    # retailer 4 overall sales
    latestretailer4overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])) & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    previousretailer4overallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])) & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    percentchangeinsalesretailer4overall = (latestretailer4overallsales - previousretailer4overallsales)/previousretailer4overallsales
    
    # retailer 4 sales
    latestretailer4sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    previousretailer4sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    percentchangeinsalesretailer4 = (latestretailer4sales - previousretailer4sales)/previousretailer4sales
    
    # overall coverage
    overallcoverage = len(set(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])),'Branch Name']))
    
    brandcoverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON'),'Branch Name']))
    percentagecoverage = brandcoverage/overallcoverage
    
    retailer1overallcoverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Customer Name']==top5retailernames[0]),'Branch Name']))
    retailer1coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[0]),'Branch Name']))
    percentageretailer1coverage = retailer1coverage/retailer1overallcoverage
    
    retailer2overallcoverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Customer Name']==top5retailernames[1]),'Branch Name']))
    retailer2coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[1]),'Branch Name']))
    percentageretailer2coverage = retailer2coverage/retailer2overallcoverage
    
    retailer3overallcoverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Customer Name']==top5retailernames[2]),'Branch Name']))
    retailer3coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[2]),'Branch Name']))
    percentageretailer3coverage = retailer3coverage/retailer3overallcoverage
    
    retailer4overallcoverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Customer Name']==top5retailernames[3]),'Branch Name']))    
    retailer4coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']=='LAMB WESTON') & (combineddata['Customer Name']==top5retailernames[3]),'Branch Name']))
    percentageretailer4coverage = retailer4coverage/retailer4overallcoverage
    
    brand0 = [latestoverallsales,percentchangeinsales,latestbrandsales,percentchangeinsalesbrand,overallcoverage,brandcoverage,percentagecoverage,
              latestretailer1overallsales,percentchangeinsalesretailer1overall,latestretailer1sales,percentchangeinsalesretailer1,retailer1overallcoverage,retailer1coverage,percentageretailer1coverage,
              latestretailer2overallsales,percentchangeinsalesretailer2overall,latestretailer2sales,percentchangeinsalesretailer2,retailer2overallcoverage,retailer2coverage,percentageretailer2coverage,
              latestretailer3overallsales,percentchangeinsalesretailer3overall,latestretailer3sales,percentchangeinsalesretailer3,retailer3overallcoverage,retailer3coverage,percentageretailer3coverage,
              latestretailer4overallsales,percentchangeinsalesretailer4overall,latestretailer4sales,percentchangeinsalesretailer4,retailer4overallcoverage,retailer4coverage,percentageretailer4coverage]
    
    other =   [top5retailernames[0],top5retailernames[1],top5retailernames[2],top5retailernames[3],top2brandnames[0],top2brandnames[1]]
    
    ### Brand 1
       
    # overall sales
    brand1latestoverallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])),'Sales Qty'])
    brand1previousoverallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])),'Sales Qty'])
    brand1percentchangeinsales = (brand1latestoverallsales - brand1previousoverallsales)/brand1previousoverallsales
    
    # brand sales
    brand1latestbrandsales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]),'Sales Qty'])
    brand1previousbrandsales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[0]),'Sales Qty'])
    brand1percentchangeinsalesbrand = (brand1latestbrandsales - brand1previousbrandsales)/brand1previousbrandsales
        
    # retailer 1 sales
    brand1latestretailer1sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    brand1previousretailer1sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    brand1percentchangeinsalesretailer1 = (brand1latestretailer1sales - brand1previousretailer1sales)/brand1previousretailer1sales
    
    # retailer 2 sales
    brand1latestretailer2sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    brand1previousretailer2sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    brand1percentchangeinsalesretailer2 = (brand1latestretailer2sales - brand1previousretailer2sales)/brand1previousretailer2sales
    
    # retailer 3 sales
    brand1latestretailer3sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    brand1previousretailer3sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    brand1percentchangeinsalesretailer3 = (brand1latestretailer3sales - brand1previousretailer3sales)/brand1previousretailer3sales
    
    # retailer 4 sales
    brand1latestretailer4sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    brand1previousretailer4sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    brand1percentchangeinsalesretailer4 = (brand1latestretailer4sales - brand1previousretailer4sales)/brand1previousretailer4sales
    
    # overall coverage
    
    brand1coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]),'Branch Name']))
    brand1percentagecoverage = brand1coverage/overallcoverage
    
    brand1retailer1coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[0]),'Branch Name']))
    brand1percentageretailer1coverage = brand1retailer1coverage/retailer1overallcoverage
    
    brand1retailer2coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[1]),'Branch Name']))
    brand1percentageretailer2coverage = brand1retailer2coverage/retailer2overallcoverage
    
    brand1retailer3coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[2]),'Branch Name']))
    brand1percentageretailer3coverage = brand1retailer3coverage/retailer3overallcoverage

    brand1retailer4coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[0]) & (combineddata['Customer Name']==top5retailernames[3]),'Branch Name']))
    brand1percentageretailer4coverage = brand1retailer4coverage/retailer4overallcoverage

    brand1 = [brand1latestoverallsales,brand1percentchangeinsales,brand1latestbrandsales,brand1percentchangeinsalesbrand,overallcoverage,brand1coverage,brand1percentagecoverage,
              latestretailer1overallsales,percentchangeinsalesretailer1overall,brand1latestretailer1sales,brand1percentchangeinsalesretailer1,retailer1overallcoverage,brand1retailer1coverage,brand1percentageretailer1coverage,
              latestretailer2overallsales,percentchangeinsalesretailer2overall,brand1latestretailer2sales,brand1percentchangeinsalesretailer2,retailer2overallcoverage,brand1retailer2coverage,brand1percentageretailer2coverage,
              latestretailer3overallsales,percentchangeinsalesretailer3overall,brand1latestretailer3sales,brand1percentchangeinsalesretailer3,retailer3overallcoverage,brand1retailer3coverage,brand1percentageretailer3coverage,
              latestretailer4overallsales,percentchangeinsalesretailer4overall,brand1latestretailer4sales,brand1percentchangeinsalesretailer4,retailer4overallcoverage,brand1retailer4coverage,brand1percentageretailer4coverage]
    
    # Brand 2
          
    # overall sales
    brand2latestoverallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year])),'Sales Qty'])
    brand2previousoverallsales = sum(combineddata.loc[combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1])),'Sales Qty'])
    brand2percentchangeinsales = (brand2latestoverallsales - brand2previousoverallsales)/brand2previousoverallsales
    
    # brand sales
    brand2latestbrandsales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]),'Sales Qty'])
    brand2previousbrandsales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[1]),'Sales Qty'])
    brand2percentchangeinsalesbrand = (brand2latestbrandsales - brand2previousbrandsales)/brand2previousbrandsales
        
    # retailer 1 sales
    brand2latestretailer1sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    brand2previousretailer1sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[0]),'Sales Qty'])
    brand2percentchangeinsalesretailer1 = (brand2latestretailer1sales - brand2previousretailer1sales)/brand2previousretailer1sales
    
    # retailer 2 sales
    brand2latestretailer2sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    brand2previousretailer2sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[1]),'Sales Qty'])
    brand2percentchangeinsalesretailer2 = (brand2latestretailer2sales - brand2previousretailer2sales)/brand2previousretailer2sales
    
    # retailer 3 sales
    brand2latestretailer3sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    brand2previousretailer3sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[2]),'Sales Qty'])
    brand2percentchangeinsalesretailer3 = (brand2latestretailer3sales - brand2previousretailer3sales)/brand2previousretailer3sales
    
    # retailer 4 sales
    brand2latestretailer4sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    brand2previousretailer4sales = sum(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year-1]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[3]),'Sales Qty'])
    brand2percentchangeinsalesretailer4 = (brand2latestretailer4sales - brand2previousretailer4sales)/brand2previousretailer4sales
    
    # overall coverage
    
    brand2coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]),'Branch Name']))
    brand2percentagecoverage = brand2coverage/overallcoverage
    
    brand2retailer1coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[0]),'Branch Name']))
    brand2percentageretailer1coverage = brand2retailer1coverage/retailer1overallcoverage
    
    brand2retailer2coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[1]),'Branch Name']))
    brand2percentageretailer2coverage = brand2retailer2coverage/retailer2overallcoverage
    
    brand2retailer3coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[2]),'Branch Name']))
    brand2percentageretailer3coverage = brand2retailer3coverage/retailer3overallcoverage

    brand2retailer4coverage = len(set(combineddata.loc[(combineddata['MonthYear'].astype(str).isin(list(listoftuples[year]))) & (combineddata['Brand Name']==top2brandnames[1]) & (combineddata['Customer Name']==top5retailernames[3]),'Branch Name']))
    brand2percentageretailer4coverage = brand2retailer4coverage/retailer4overallcoverage

    brand2 = [brand2latestoverallsales,brand2percentchangeinsales,brand2latestbrandsales,brand2percentchangeinsalesbrand,overallcoverage,brand2coverage,brand2percentagecoverage,
              latestretailer1overallsales,percentchangeinsalesretailer1overall,brand2latestretailer1sales,brand2percentchangeinsalesretailer1,retailer1overallcoverage,brand2retailer1coverage,brand2percentageretailer1coverage,
              latestretailer2overallsales,percentchangeinsalesretailer2overall,brand2latestretailer2sales,brand2percentchangeinsalesretailer2,retailer2overallcoverage,brand2retailer2coverage,brand2percentageretailer2coverage,
              latestretailer3overallsales,percentchangeinsalesretailer3overall,brand2latestretailer3sales,brand2percentchangeinsalesretailer3,retailer3overallcoverage,brand2retailer3coverage,brand2percentageretailer3coverage,
              latestretailer4overallsales,percentchangeinsalesretailer4overall,brand2latestretailer4sales,brand2percentchangeinsalesretailer4,retailer4overallcoverage,brand2retailer4coverage,brand2percentageretailer4coverage]

    return(brand0,brand1,brand2,other)
    
### AskAI backend
def texttopandas(combineddata,instruction: str):
    
    # Has different function within this
    # 1. Starts with identifying if the user has requested for a table or not
    # 2. Follows by getting the pandas code
    # 3. Executing the code generated
    # 4. Summarizing the results generated from a data frame (skipped if the request is a table)
    # This is followed by the current function's code where these functions are called
   
    ## identifying if a table is asked explicitly
    def user_wants_table(instruction: str) -> bool:
       
        table_keywords = ["table", "tabular", "dataframe", "show data", "display data", "display table"]
        instruction_lower = instruction.lower()
        return any(kw in instruction_lower for kw in table_keywords)
    
    ## Generating Pandas Code
    def get_pandas_code(combineddata,instruction: str) -> str:
        
        prompt = f"""
            You are a helpful assistant that only returns executable pandas code using a DataFrame called combineddata.
            The DataFrame has the following columns: {', '.join(combineddata.columns)} and the data type of these columns are {combineddata.dtypes}. Use this when interpreting instruction and creating pandas code
            In terms of the different values that are available under each column, the same is provided to match any misreprsentation by the user in the instruction. For e.g., the user may say French Fries but the value in the subcategory column is '_FRENCH FRIES'. Same can happen across other columns as well. 
            Hence the below list of unique values across important columns and alternate names for columns is provided for your reference:
            1. Subcategory: {set(combineddata['Subcategory'])}
            2. Customer Name (also referred as retailer by the user):  {set(combineddata['Customer Name'])}
            3. Branch Name (also referred as store by the user): {set(combineddata['Branch Name'])}
            4. Brand Name: {set(combineddata['Brand Name'])}
            5. Year: {set(combineddata['Year2'])}
            6. MonthYearShortForm: {set(combineddata['MonthYearShortForm'])}
            
            Separately, different word interpretations are 
            1. Sales AED: can also be referred as value by user
            2. Sales Qty: can also be referred as volume by user
            
            For date range DO NOT use .between especially when the date column is strings e.g., Jan-2024 etc.,
            Now below is the user's question
            
            Instruction: {instruction}
            Return ONLY the pandas expression that evaluates to a DataFrame. Do NOT assign it to a variable.
            Do NOT include markdown code block syntax like ```python or any other syntax.
            """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python assistant who only returns pandas code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return(response.choices[0].message.content.strip())
    
    ## Execute code - #eval helps in evaluating a python expression which
    #  takes the code / practically a string as the 1st input
    #  followed by the dataframe, variable on which the code can be evaluated
    def run_code(code: str, local_vars: dict) -> pd.DataFrame:
        try:
            result = eval(code, globals(), local_vars)
            return result
        except Exception as e:
            return f"Error running code: {e}"
    
    ## Generate summary in natural language (if the request is not a table alone)
    def summarize_in_natural_language(df, original_instruction: str) -> str:
        
        # handling different data types that arise from executing the pandas code
        if isinstance(df,pd.DataFrame):
            df = df.to_string(index=False)
        elif isinstance(df,pd.Series):
            df = df.to_string()
        else:
            df = df
        
        # passing the data to OpenAI API
        prompt = f"""
        You are a data analyst. Based on the user's instruction and the following DataFrame result, write a brief natural language summary. If the no of entries in the resultant dataframe is high, feel free for e.g., to order and summarize Top 5 and give an overall picture
        User instruction: "{original_instruction}"
        Data: {df}
        
        Summary:
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analyst that summarizes pandas dataframes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return(response.choices[0].message.content.strip())

    ## End-to-end Execution - calling above functions       
    code = get_pandas_code(combineddata,instruction)
    print(code)
    result = run_code(code, {"combineddatareq": combineddata})
    
    # Checking if the user wants a table else it is sent into summarization module   
    if isinstance(result, (pd.DataFrame,pd.Series)):
        if user_wants_table(instruction):
            # Handling for pd.Series output from run_code eval function
            if isinstance(result, pd.Series):
                df = result.to_frame()
                return([None,df])
            # Handles pd.DataFrame by default
            return([None,result])
        else:
            summary = summarize_in_natural_language(result, instruction)
            return([summary,None])
    # Other object types (i.e., other than dataframe, series)
    elif isinstance(result,str):
        summary = summarize_in_natural_language(result, instruction)
        return([summary,None])
    elif isinstance(result,int):
        summary = summarize_in_natural_language(result, instruction)
        return([summary,None])
    elif isinstance(result,float):
        summary = summarize_in_natural_language(result, instruction)
        return([summary,None])
    else:
        return([str(result),None])
    
### UI & getting into the flow

### State initializations

## Defining category session state for category as empty to start with
if 'selected_country' not in st.session_state:   
    st.session_state.selected_country = []
if 'selected_retailer' not in st.session_state:
    st.session_state.selected_retailer = []
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = ''
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = ''
if 'category_content_generated' not in st.session_state:
    st.session_state.category_content_generated = False

# Data storage for category graphs
if 'category_volume_yearly_data' not in st.session_state:
    st.session_state.category_volume_yearly_data = None
if 'category_volume_monthly_data' not in st.session_state:
    st.session_state.category_volume_monthly_data = None
if 'category_value_yearly_data' not in st.session_state:
    st.session_state.category_value_yearly_data = None
if 'category_value_monthly_data' not in st.session_state:
    st.session_state.category_value_monthly_data = None
    
## Defining subcategory related session state as empty to start with
if 'selected_subcatcountry' not in st.session_state:   
    st.session_state.selected_subcatcountry = []
if 'selected_subcatretailer' not in st.session_state:
    st.session_state.selected_subcatretailer = []
if 'selected_subcategory' not in st.session_state:
    st.session_state.selected_subcategory = []
if 'selected_subcatmode' not in st.session_state:
    st.session_state.selected_subcatmode = ''
if 'selected_subcatmonth' not in st.session_state:
    st.session_state.selected_subcatmonth = ''
if 'subcategory_content_generated' not in st.session_state:
    st.session_state.subcategory_content_generated = False

# Data storage for subcategory graphs
if 'subcategory_vol_yearly_data' not in st.session_state:
    st.session_state.subcategory_vol_yearly_data = None
if 'subcategory_vol_monthly_data' not in st.session_state:
    st.session_state.subcategory_vol_monthly_data = None
if 'subcategory_val_yearly_data' not in st.session_state:
    st.session_state.subcategory_val_yearly_data = None
if 'subcategory_val_monthly_data' not in st.session_state:
    st.session_state.subcategory_val_monthly_data = None
    
## Defining brand related session state as empty to start with
if 'selected_brand' not in st.session_state:   
    st.session_state.selected_brand = []
if 'brand_selected_retailer' not in st.session_state:
    st.session_state.brand_selected_retailer = []
if 'brand_selected_subcategory' not in st.session_state:
    st.session_state.brand_selected_subcategory = []
if 'brand_selected_mode' not in st.session_state:
    st.session_state.brand_selected_mode = ''
if 'brand_selected_month' not in st.session_state:
    st.session_state.brand_selected_month = ''
if 'brand_selected_outlet' not in st.session_state:
    st.session_state.brand_selected_outlet = ''
if 'brand_selected_region' not in st.session_state:
    st.session_state.brand_selected_region = ''   
if 'brand_content_generated' not in st.session_state:
    st.session_state.brand_content_generated = False
    
# Data storage for brand graphs & tables
if 'brand_matytd' not in st.session_state:
    st.session_state.brand_matytd = None
if 'brand_threemonth' not in st.session_state:
    st.session_state.brand_threemonth = None
if 'brand_currentmonth' not in st.session_state:
    st.session_state.brand_currentmonth = None
if 'brand_matytd_table' not in st.session_state:
    st.session_state.brand_matytd_table = None
if 'brand_matytd_sharetable' not in st.session_state:
    st.session_state.brand_matytd_sharetable = None
if 'brand_threemonth_table' not in st.session_state:
    st.session_state.brand_threemonth_table = None
if 'brand_currentmonth_table' not in st.session_state:
    st.session_state.brand_currentmonth_table = None
    
## Creating the tabs here
tab0, tab1, tab2, tab3, tab4 = st.tabs(["AskAI","Monitor My Business","Category Performance","Sub Category Performance","Brand Performance"])

## AI chatbot
with tab0:
    
    st.title("Ask your AI Buddy!")
    
    ## Chat window
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Initialize chat history
    for chat in st.session_state.history:
       st.markdown(f"**🧑 You:** {chat['instruction']}")
       # For a dataframe display as the user requested so, then it moves to a different section
       if chat.get("df") is not None:
           col1, col2, col3 = st.columns([1, 2, 1])  # Center column is wider
           with col1:
               st.markdown(f"**🗣️ Assistant:**")
               st.dataframe(chat["df"])
       # Summarization
       elif chat.get("summary"):
           st.markdown(f"**🗣️ Assistant:** {chat['summary']}")
       st.markdown("---")
    
    # Data structure expander
    with st.expander("🔍 Explore Data Structure"):
        col_data = []
        for col in combineddata.columns:
            dtype = combineddata[col].dtype
            unique_vals = combineddata[col].dropna().unique()
            sample_vals = ", ".join(map(str, unique_vals[:5]))
            if len(unique_vals) > 5:
                sample_vals += "..."
            # Creating a dictionary where the keys could be the column names and a dictionary for each column is fed/appended into a list
            col_data.append({
                "Column": col,
                "Unique Values": combineddata[col].nunique(),
                "Sample Values": sample_vals
            })
        # Displaying the same
        col_df = pd.DataFrame(col_data)
        st.dataframe(col_df, use_container_width=True)
        
    # Followed by the query form to obtain input    
    with st.form("query_form", clear_on_submit=True):
        instruction = st.text_input("Ask Anything:", placeholder="e.g., Total sales by Category for 2024")
        submitted = st.form_submit_button("Submit")
        if submitted and instruction:
            with st.spinner("Thinking..."):
                
                reqoutput = texttopandas(combineddata,instruction)
                  
                # Save this run to history
                st.session_state.history.append({
                    "instruction": instruction,
                    "summary": reqoutput[0],
                    "df": reqoutput[1]  # could be None
                })
                
                # Force rerun to push new result to top
                st.rerun()
with tab1:
    st.title("Scorecard")
    
    # Define custom CSS for black theme and card-like structure
    st.markdown("""
        <style>
            .card {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 10px;
                margin: 5px;
                text-align: left;
                width: 100%;
            }
            .card h4 {
                color: #ffffff;
                margin: 0 0 8px 0;
                font-size: 16px;
            }
            .card .value {
                font-size: 18px;
                font-weight: bold;
                margin: 4px 0;
            }
            .card .change {
                font-size: 14px;
            }
            .positive { color: #4CAF50; }
            .negative { color: #F44336; }
        </style>
    """, unsafe_allow_html=True)

    # Calling backend function
    kpival = kpis(combineddata)    
    with st.expander("💼 Competitor Brand Section", expanded=True):
        table1, table2, table3, table4 = st.columns(4)

        def render_table_html(cat_name, brand_name, cat_vals, brand_vals, cov_vals):
            def format_delta(val):
                arrow = "▲" if val > 0 else "▼" if val < 0 else ""
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f'<span style="color: {color}; font-weight: bold;">{arrow} {val*100:.1f}%</span>'
            return f"""
            <div style="border: 1px solid #ccc; border-radius: 6px; padding: 0 10px; background-color: #ffffff; color: #000000; font-family: Arial;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; border: none; margin: 0;">
                    <thead>
                        <tr style="border: none;">
                            <th style="text-align: left; padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">{cat_name}</th>
                            <th style="text-align: left; padding: 6px 4px;text-align: center; vertical-align: middle; border: none;">{brand_name}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;">Category Sales</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none;">Sales</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; font-size: 16px; border-top: none; border-bottom: none;">{cat_vals[0]/1000:.1f}K</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none; font-size: 16px;">{brand_vals[0]/1000:.1f}K</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">{format_delta(cat_vals[1])}</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{format_delta(brand_vals[1])}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">Coverage</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none;">Coverage</td>
                        </tr>   
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;">{cov_vals[0]:,.0f}</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{cov_vals[1]:,.0f}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;"></td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{format_delta(cov_vals[2])} ND</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """


        with table1:
            st.markdown(render_table_html(
                "Category",
                "Lamb Weston",
                cat_vals=[kpival[0][0], kpival[0][1]],
                brand_vals=[kpival[0][2], kpival[0][3]],
                cov_vals=[kpival[0][4], kpival[0][5], kpival[0][6]]
            ), unsafe_allow_html=True)

        with table2:
            st.markdown(render_table_html(
                f"{kpival[3][0]} - Category",
                f"{kpival[3][0]} - Lamb Weston",
                cat_vals=[kpival[0][7], kpival[0][8]],
                brand_vals=[kpival[0][9], kpival[0][10]],
                cov_vals=[kpival[0][11], kpival[0][12], kpival[0][13]]
            ), unsafe_allow_html=True)

        with table3:
            st.markdown(render_table_html(
                f"{kpival[3][1]} - Category",
                f"{kpival[3][1]} - Lamb Weston",
                cat_vals=[kpival[0][14], kpival[0][15]],
                brand_vals=[kpival[0][16], kpival[0][17]],
                cov_vals=[kpival[0][18], kpival[0][19], kpival[0][20]]
            ), unsafe_allow_html=True)

        with table4:
            st.markdown(render_table_html(
                f"{kpival[3][2]} - Category",
                f"{kpival[3][2]} - Lamb Weston",
                cat_vals=[kpival[0][21], kpival[0][22]],
                brand_vals=[kpival[0][23], kpival[0][24]],
                cov_vals=[kpival[0][25], kpival[0][26], kpival[0][27]]
            ), unsafe_allow_html=True)

    
    with st.expander(f"💼 Competitor {kpival[3][4]} Brand Section", expanded=True):
        table1, table2, table3, table4 = st.columns(4)

        def render_table_html(cat_name, brand_name, cat_vals, brand_vals, cov_vals):
            def format_delta(val):
                arrow = "▲" if val > 0 else "▼" if val < 0 else ""
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f'<span style="color: {color}; font-weight: bold;">{arrow} {val*100:.1f}%</span>'
            return f"""
            <div style="border: 1px solid #ccc; border-radius: 6px; padding: 0 10px; background-color: #ffffff; color: #000000; font-family: Arial;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; border: none; margin: 0;">
                    <thead>
                        <tr style="border: none;">
                            <th style="text-align: left; padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">{cat_name}</th>
                            <th style="text-align: left; padding: 6px 4px;text-align: center; vertical-align: middle; border: none;">{brand_name}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;">Category Sales</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none;">Sales</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; font-size: 16px; border-top: none; border-bottom: none;">{cat_vals[0]/1000:.1f}K</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none; font-size: 16px;">{brand_vals[0]/1000:.1f}K</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">{format_delta(cat_vals[1])}</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{format_delta(brand_vals[1])}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">Coverage</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none;">Coverage</td>
                        </tr>   
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;">{cov_vals[0]:,.0f}</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{cov_vals[1]:,.0f}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;"></td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{format_delta(cov_vals[2])} ND</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """

        with table1:
            st.markdown(render_table_html(
                "Category",
                "Lamb Weston",
                cat_vals=[kpival[1][0], kpival[1][1]],
                brand_vals=[kpival[1][2], kpival[1][3]],
                cov_vals=[kpival[1][4], kpival[1][5], kpival[1][6]]
            ), unsafe_allow_html=True)

        with table2:
            st.markdown(render_table_html(
                f"{kpival[3][0]} - Category",
                f"{kpival[3][0]} - Lamb Weston",
                cat_vals=[kpival[1][7], kpival[1][8]],
                brand_vals=[kpival[1][9], kpival[1][10]],
                cov_vals=[kpival[1][11], kpival[1][12], kpival[1][13]]
            ), unsafe_allow_html=True)

        with table3:
            st.markdown(render_table_html(
                f"{kpival[3][1]} - Category",
                f"{kpival[3][1]} - Lamb Weston",
                cat_vals=[kpival[1][14], kpival[1][15]],
                brand_vals=[kpival[1][16], kpival[1][17]],
                cov_vals=[kpival[1][18], kpival[1][19], kpival[1][20]]
            ), unsafe_allow_html=True)

        with table4:
            st.markdown(render_table_html(
                f"{kpival[3][2]} - Category",
                f"{kpival[3][2]} - Lamb Weston",
                cat_vals=[kpival[1][21], kpival[1][22]],
                brand_vals=[kpival[1][23], kpival[1][24]],
                cov_vals=[kpival[1][25], kpival[1][26], kpival[1][27]]
            ), unsafe_allow_html=True)
    with st.expander(f"💼 Competitor {kpival[3][5]} Brand Section", expanded=True):
        table1, table2, table3, table4 = st.columns(4)

        def render_table_html(cat_name, brand_name, cat_vals, brand_vals, cov_vals):
            def format_delta(val):
                arrow = "▲" if val > 0 else "▼" if val < 0 else ""
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f'<span style="color: {color}; font-weight: bold;">{arrow} {val*100:.1f}%</span>'
            return f"""
            <div style="border: 1px solid #ccc; border-radius: 6px; padding: 0 10px; background-color: #ffffff; color: #000000; font-family: Arial;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; border: none; margin: 0;">
                    <thead>
                        <tr style="border: none;">
                            <th style="text-align: left; padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">{cat_name}</th>
                            <th style="text-align: left; padding: 6px 4px;text-align: center; vertical-align: middle; border: none;">{brand_name}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;">Category Sales</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none;">Sales</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; font-size: 16px; border-top: none; border-bottom: none;">{cat_vals[0]/1000:.1f}K</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none; font-size: 16px;">{brand_vals[0]/1000:.1f}K</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">{format_delta(cat_vals[1])}</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{format_delta(brand_vals[1])}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc; border-left: none; border-top: none; border-bottom: none;">Coverage</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border: none;">Coverage</td>
                        </tr>   
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;">{cov_vals[0]:,.0f}</td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{cov_vals[1]:,.0f}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle; border-right: 1px solid #ccc;border-left: none; border-top: none; border-bottom: none;"></td>
                            <td style="padding: 6px 4px; text-align: center;vertical-align: middle;border: none;">{format_delta(cov_vals[2])} ND</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """
        with table1:
            st.markdown(render_table_html(
                "Category",
                "Lamb Weston",
                cat_vals=[kpival[2][0], kpival[2][1]],
                brand_vals=[kpival[2][2], kpival[2][3]],
                cov_vals=[kpival[2][4], kpival[2][5], kpival[2][6]]
            ), unsafe_allow_html=True)

        with table2:
            st.markdown(render_table_html(
                f"{kpival[3][0]} - Category",
                f"{kpival[3][0]} - Lamb Weston",
                cat_vals=[kpival[2][7], kpival[2][8]],
                brand_vals=[kpival[2][9], kpival[2][10]],
                cov_vals=[kpival[2][11], kpival[2][12], kpival[2][13]]
            ), unsafe_allow_html=True)

        with table3:
            st.markdown(render_table_html(
                f"{kpival[3][1]} - Category",
                f"{kpival[3][1]} - Lamb Weston",
                cat_vals=[kpival[2][14], kpival[2][15]],
                brand_vals=[kpival[2][16], kpival[2][17]],
                cov_vals=[kpival[2][18], kpival[2][19], kpival[2][20]]
            ), unsafe_allow_html=True)

        with table4:
            st.markdown(render_table_html(
                f"{kpival[3][2]} - Category",
                f"{kpival[3][2]} - Lamb Weston",
                cat_vals=[kpival[2][21], kpival[2][22]],
                brand_vals=[kpival[2][23], kpival[2][24]],
                cov_vals=[kpival[2][25], kpival[2][26], kpival[2][27]]
            ), unsafe_allow_html=True)
    
with tab2: #category dropdown
               
    ### Obtaining inputs
    st.title("Category Performance")
    st.subheader("Slicers",divider="gray")
    
    ### Creating containers for obtaining graphs
    with st.container(border=True):
        
        # Create 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            countrylist = st.multiselect('Country',allcountry,key='category_country_selector')
        with col2:
            retailerlist = st.multiselect('Retailer',allretailer,key='category_retailer_selector')
        with col3:
            modelist = st.selectbox('YTD/MAT',['YTD','MAT'], key='category_mode_selector')
        with col4:
            monthlist = st.selectbox('Month',sortedmonths,index=len(sortedmonths) - 1,key='category_month_selector')
    
    ### Generating required data and graphs
    
    ## Checking for pass through & to contain regeneration - making the tab delinked/independent
    selections_changed = (countrylist != st.session_state.selected_country or retailerlist != st.session_state.selected_retailer or modelist != st.session_state.selected_mode or monthlist != st.session_state.selected_month)    
    has_valid_selections = (len(countrylist) > 0 and len(retailerlist) > 0 and modelist and monthlist)
    
    if selections_changed and has_valid_selections:
              
        ## Initializing state if it passes through this        
        st.session_state.selected_country = countrylist
        st.session_state.selected_retailer = retailerlist
        st.session_state.selected_mode = modelist
        st.session_state.selected_month = monthlist
        
        # Generate and store volume data
        reqvol = volume(combineddata, retailerlist, countrylist, modelist, monthlist, retailer, country)
        st.session_state.category_volume_yearly_data = reqvol[0]
        st.session_state.category_volume_monthly_data = reqvol[1]
       
        # Generate and store value data
        reqval = value(combineddata, retailerlist, countrylist, modelist, monthlist, retailer, country)
        st.session_state.category_value_yearly_data = reqval[0]
        st.session_state.category_value_monthly_data = reqval[1]
       
        # AI generation — only run once
        catai_vol_output = categoryaianalysis(reqvol[0], reqvol[1], 'Volume', retailerlist, countrylist, modelist, monthlist)
        catai_val_output = categoryaianalysis(reqval[0], reqval[1], 'Value', retailerlist, countrylist, modelist, monthlist)
        # Save to session
        st.session_state.category_ai_vol_output = catai_vol_output
        st.session_state.category_ai_val_output = catai_val_output
        
        # Mark that content has been generated
        st.session_state.category_content_generated = True
           
    # Display content if it has been generated (not just when selections change)
    if st.session_state.category_content_generated and has_valid_selections:
        ## Volume    
        st.subheader("Volume Analysis",divider="gray")
        
        # Graphs
        with st.container():
            yearlyvolgraph, monthvolgraph = st.columns([1,2])
            with yearlyvolgraph:
                
                yearly_vol_data = st.session_state.category_volume_yearly_data
                fig = px.bar(yearly_vol_data.astype(str).sort_values('Years'),x='Years',y='Sales Qty',text=(yearly_vol_data['Sales Qty']/1e6).map(lambda x: f'{x:.2f}'),title='Yearly',labels={'Sales Qty': 'Volume'},color_discrete_sequence=['skyblue'],hover_data={'Years': True, 'Sales Qty': ':.2f'})
                fig.update_traces(texttemplate='%{text}', textposition='outside',width=0.4,hovertemplate='<b>Year:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>')
                fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Years', tickfont=dict(size=12),type='category'),yaxis=dict(title='Volume (Mn)', tickfont=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True,key='category_vol_year')
            
            with monthvolgraph:
                monthly_vol_data = st.session_state.category_volume_monthly_data
                fig = px.bar(monthly_vol_data,x='Months',y='Sales Qty',text=(monthly_vol_data['Sales Qty']/1000).round(2),title='Monthly',labels={'Sales Qty': 'Volume'},color_discrete_sequence=['orange'])
                fig.update_traces(texttemplate='%{text}',textposition='outside',hovertemplate='<b>Month:</b> %{x}<br><b>Volume:</b> %{y}<extra></extra>')
                fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Months', tickfont=dict(size=12),tickmode='array',tickvals=monthly_vol_data['Months'],ticktext=monthly_vol_data['Months'].dt.strftime('%b-%y')),yaxis=dict(title="Volume('000)", tickfont=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True,key='category_vol_month')
        
        # AI Content
        with st.container(border=True):
             st.markdown(st.session_state.category_ai_vol_output)
                                        
        ## Value
        st.subheader("Value Analysis",divider="gray")
        
        # Graphs
        with st.container():
            yearlyvalgraph, monthvalgraph = st.columns([1,2])
            with yearlyvalgraph:
                
                yearly_val_data = st.session_state.category_value_yearly_data
                fig = px.bar(yearly_val_data.astype(str).sort_values('Years'),x='Years',y='Sales AED',text=(yearly_val_data['Sales AED']/1e6).map(lambda x: f'{x:.2f}'),title='Yearly',labels={'Sales AED': 'Value'},color_discrete_sequence=['skyblue'])
                fig.update_traces(texttemplate='%{text}', textposition='outside',width=0.4,hovertemplate='<b>Year:</b> %{x}<br><b>Value:</b> %{y:,.0f} AED<extra></extra>')
                fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Years', tickfont=dict(size=12),type='category'),yaxis=dict(title='Value (Mn)', tickfont=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True,key='category_val_year')
            
            with monthvalgraph:
                monthly_val_data = st.session_state.category_value_monthly_data
                fig = px.bar(monthly_val_data,x='Months',y=monthly_val_data['Sales AED']/1e6,text=(monthly_val_data['Sales AED'] / 1e6).map(lambda x: f'{x:.2f}'),title='Monthly',labels={'Sales AED': 'Value (Mn AED)'},color_discrete_sequence=['orange'], custom_data=[monthly_val_data['Sales AED']])
                fig.update_traces(texttemplate='%{text}',textposition='outside',hovertemplate='<b>Month:</b> %{x}<br><b>Value:</b> %{customdata[0]:,.0f} AED<extra></extra>')
                fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Months', tickfont=dict(size=12),tickmode='array',tickvals=monthly_val_data['Months'],ticktext=monthly_val_data['Months'].dt.strftime('%b-%y')),yaxis=dict(title='Value (Mn)',tickfont=dict(size=12),tickformat=".0f",ticksuffix="M"))
                st.plotly_chart(fig, use_container_width=True,key='category_val_month')
        
        # AI Content
        with st.container(border=True):
            st.markdown(st.session_state.category_ai_val_output)
        
with tab3: ## Sub category
    
    ### Obtaining inputs
    st.title("Sub Category Performance")
    st.subheader("Slicers",divider="gray")
    
    ### Creating containers for obtaining inputs
    with st.container(border=True):
        
        ## Create 4 columns
        col5, col6, col7, col8, col9 = st.columns(5)
        with col5:
            subcatcountrylist = st.multiselect('Country',allsubcatcountry,key='subcategory_country_selector')
        with col6:
            subcatretailerlist = st.multiselect('Retailer',allsubcatretailer,key='subcategory_retailer_selector')
        with col7:
            subcategorylist = st.multiselect('Sub Category',allsubcategory)
        with col8:
            subcatmodelist = st.selectbox('YTD/MAT',['YTD','MAT'],key='subcategory_mode_selector')
        with col9:
            subcatmonthlist = st.selectbox('Month',subcatsortedmonths,index=len(subcatsortedmonths) - 1,key='subcategory_month_selector')
   
    ### Generating required data and graphs
    
    ## Checking for pass through & to contain regenration - making the tab delinked/independent
    subcatselections_changed = (subcatcountrylist != st.session_state.selected_subcatcountry or subcatretailerlist != st.session_state.selected_subcatretailer or subcategorylist != st.session_state.selected_subcategory or subcatmodelist != st.session_state.selected_subcatmode or subcatmonthlist != st.session_state.selected_subcatmonth)
    subcathas_valid_selections = (len(subcatcountrylist) > 0 and len(subcatretailerlist) > 0 and len(subcategorylist)>0 and subcatmodelist and subcatmonthlist)
    
    if subcatselections_changed and subcathas_valid_selections:
              
        ## Initializing state if it passes through this
        st.session_state.selected_subcatcountry = subcatcountrylist
        st.session_state.selected_subcatretailer = subcatretailerlist
        st.session_state.selected_subcategory = subcategorylist
        st.session_state.selected_subcatmode = subcatmodelist
        st.session_state.selected_subcatmonth = subcatmonthlist
        
        # Generate and store data for subcategory
        # Volume data
        subcatreqvol = subcategoryvolume(combineddata, subcatretailerlist, subcatcountrylist, subcategorylist, subcatmodelist, subcatmonthlist, subcatretailer, subcatcountry, subcategory)
        st.session_state.subcategory_vol_yearly_data = subcatreqvol[0]
        st.session_state.subcategory_vol_monthly_data = subcatreqvol[1]
        
        # Value data
        subcatreqval = subcategoryvalue(combineddata, subcatretailerlist, subcatcountrylist, subcategorylist, subcatmodelist, subcatmonthlist, subcatretailer, subcatcountry, subcategory)
        st.session_state.subcategory_val_yearly_data = subcatreqval[0]
        st.session_state.subcategory_val_monthly_data = subcatreqval[1]
    
        # AI generation — only run once
        ai_vol_output = subcategoryaianalysis(subcatreqvol[0], subcatreqvol[1], 'Volume', subcatretailerlist, subcatcountrylist, subcategorylist, subcatmodelist, subcatmonthlist)
        ai_val_output = subcategoryaianalysis(subcatreqval[0], subcatreqval[1], 'Value', subcatretailerlist, subcatcountrylist, subcategorylist, subcatmodelist, subcatmonthlist)
        # Save to session
        st.session_state.subcategory_ai_vol_output = ai_vol_output
        st.session_state.subcategory_ai_val_output = ai_val_output
            
        # Mark that subcategory content has been generated
        st.session_state.subcategory_content_generated = True
        
    if st.session_state.subcategory_content_generated and subcathas_valid_selections:

        ## Volume    
        st.subheader("Volume Analysis",divider="gray")
        
        # Graphs
        with st.container():
            subcatyearlyvolgraph, subcatmonthvolgraph = st.columns([1,2])
            with subcatyearlyvolgraph:
                # Data Preparatiom
                subcatreqvolyear = st.session_state.subcategory_vol_yearly_data
                subcatreqvolyear_melted = subcatreqvolyear.melt(id_vars='Years', value_vars=subcatreqvolyear.columns[1:], var_name='Sub Category', value_name='Volume')
                subcatreqvolyear_melted = subcatreqvolyear_melted.groupby(['Years', 'Sub Category'], as_index=False)['Volume'].sum()
                subcatreqvolyear_melted = subcatreqvolyear_melted.sort_values(by=['Years','Volume'])
                
                # Plot Details
                color_map = {
                    '_FRENCH FRIES': '#4e79a7',   # blue
                    '_ONION RINGS': '#e15759',    # red
                    '_HASHBROWNS': '#76b7b2',     # teal
                    '_POTATOES': '#59a14f',       # green
                }
                fig = px.bar(subcatreqvolyear_melted,x='Years',y='Volume',color = 'Sub Category',barmode='stack',text=(subcatreqvolyear_melted['Volume']/1e6).round(2),title='Yearly',color_discrete_map=color_map,hover_data={'Years': True, 'Volume': ':.2f', 'Sub Category': False})
                fig.update_traces(texttemplate='%{text}',textposition='inside',hovertemplate='<b>Year:</b> %{x}<br><b>Volume:</b> %{y:,.2f}<extra></extra>')
                st.plotly_chart(fig, use_container_width=True,key='subcategory_vol_year')
    
            with subcatmonthvolgraph:
                # Data Preparation
                subcatreqvolmonth = st.session_state.subcategory_vol_monthly_data
                subcatreqvolmonth_melted = subcatreqvolmonth.melt(id_vars='Months', value_vars=subcatreqvolyear.columns[1:], var_name='Sub Category', value_name='Volume')
                subcatreqvolmonth_melted = subcatreqvolmonth_melted.sort_values(by=['Months','Volume'],ascending=[True,True])
                
                # Plot Details
                fig = px.bar(subcatreqvolmonth_melted,x='Months',y='Volume',color='Sub Category',text=(subcatreqvolmonth_melted['Volume']/1000).round(2),title='Monthly',color_discrete_map=color_map)
                fig.update_traces(texttemplate='%{text}',textposition='inside',hovertemplate='<b>Month:</b> %{x}<br><b>Volume:</b> %{y}<extra></extra>')
                fig.update_layout(barmode='stack',plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Months', tickfont=dict(size=12),tickmode='array',tickvals=subcatreqvolmonth_melted['Months'],ticktext=subcatreqvolmonth_melted['Months'].dt.strftime('%b-%y')),yaxis=dict(title="Volume('000)", tickfont=dict(size=12),))
                st.plotly_chart(fig, use_container_width=True,key='subcategory_vol_month')
       
          # AI Content
        with st.container(border=True):
            st.markdown(st.session_state.subcategory_ai_vol_output)
                                        
        ## Value
        st.subheader("Value Analysis",divider="gray")
        
        # Graohs
        with st.container():
            subcatyearlyvalgraph, subcatmonthvalgraph = st.columns([1,2])
            with subcatyearlyvalgraph:
                # Data Preparation
                subcatreqvalyear = st.session_state.subcategory_val_yearly_data
                subcatreqvalyear_melted = subcatreqvalyear.melt(id_vars='Years', value_vars=subcatreqvalyear.columns[1:], var_name='Sub Category', value_name='Value')
                subcatreqvalyear_melted = subcatreqvalyear_melted.groupby(['Years', 'Sub Category'], as_index=False)['Value'].sum()
                subcatreqvalyear_melted = subcatreqvalyear_melted.sort_values(by=['Years','Value'])
                
                # Plot details
                fig = px.bar(subcatreqvalyear_melted.sort_values(by=['Years','Value']),x='Years',y='Value',color='Sub Category',text=(subcatreqvalyear_melted['Value']/1e6).map(lambda x: f'{x:.2f}'),title='Yearly',color_discrete_map=color_map)
                fig.update_traces(texttemplate='%{text}', textposition='inside',width=0.4,hovertemplate='<b>Year:</b> %{x}<br><b>Value:</b> %{y:,.0f} AED<extra></extra>')
                fig.update_layout(barmode='stack',plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Years', tickfont=dict(size=12),type='category'),yaxis=dict(title='Value (Mn)', tickfont=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True,key='subcategory_val_year')
            
            with subcatmonthvalgraph:
                # Data Preparation
                subcatreqvalmonth = st.session_state.subcategory_val_monthly_data
                subcatreqvalmonth_melted = subcatreqvalmonth.melt(id_vars='Months', value_vars=subcatreqvalyear.columns[1:], var_name='Sub Category', value_name='Value')
                subcatreqvalmonth_melted = subcatreqvalmonth_melted.sort_values(by=['Months','Value'],ascending=[True,True])
                # Plot Details
                fig = px.bar(subcatreqvalmonth_melted,x='Months',y=subcatreqvalmonth_melted['Value']/1e6,color='Sub Category',text=(subcatreqvalmonth_melted['Value'] / 1e6).map(lambda x: f'{x:.2f}'),title='Monthly',color_discrete_map=color_map, custom_data=[subcatreqvalmonth_melted['Value']])
                fig.update_traces(texttemplate='%{text}',textposition='inside',hovertemplate='<b>Month:</b> %{x}<br><b>Value:</b> %{customdata[0]:,.0f} AED<extra></extra>')
                fig.update_layout(barmode='stack',plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font_size=18,xaxis=dict(title='Months', tickfont=dict(size=12),tickmode='array',tickvals=subcatreqvalmonth_melted['Months'],ticktext=subcatreqvalmonth_melted['Months'].dt.strftime('%b-%y')),yaxis=dict(title='Value (Mn)',tickfont=dict(size=12),tickformat=".0f",ticksuffix="M"))
                st.plotly_chart(fig, use_container_width=True,key='subcategory_val_month')
        
        # AI Content
        with st.container(border=True):
            st.markdown(st.session_state.subcategory_ai_val_output)

with tab4: # brand
    
    ### Obtaining inputs
    st.title("Brand Performance")
    st.subheader("Slicers",divider="gray")
    
    ### Creating containers for obtaining inputs
    with st.container(border=True):
        
        ## Create 4 columns
        col10, col11, col12, col13 = st.columns(4)
        with col10:
            brandlist = st.multiselect('Brand',allbrand,key='brand_selector')
        with col11:
            brandretailerlist = st.multiselect('Retailer',allbrandretailer,key='brand_retailer_selector')
        with col12:
            brandsubcategorylist = st.multiselect('Sub Category',allbrandsubcategory,key='brand_subcategory_selector')
        with col13:
            brandoutletlist = st.multiselect('Outlet',allbrandoutlet,key='brand_outlet_selector')
        
        ## Create next 3 columns
        col15, col16, col17 = st.columns(3)
        with col15:
            brandregionlist = st.multiselect('Region',['All','Dubai','Abu Dhabi','Sharjah','Other Emirates'],key='brand_region_selector')
        with col16:
            brandmodelist = st.selectbox('YTD/MAT',['YTD','MAT'],key='brand_mode_selector')
        with col17:
            brandmonthlist = st.selectbox('Month',brandsortedmonths,index=len(brandsortedmonths) - 1,key='brand_month_selector')
    
    ### Generating required data and graphs
    
    ## Checking for pass through & to contain regenration - making the tab delinked/independent
    brandselections_changed = (brandlist != st.session_state.selected_brand or brandretailerlist != st.session_state.brand_selected_retailer or brandmodelist != st.session_state.brand_selected_mode or brandmonthlist != st.session_state.brand_selected_month or brandsubcategorylist != st.session_state.brand_selected_subcategory or brandoutletlist != st.session_state.brand_selected_outlet or brandregionlist != st.session_state.brand_selected_region)
    brandhas_valid_selections = (len(brandlist) > 0 and len(brandretailerlist) > 0 and len(brandsubcategorylist)>0 and len(brandoutletlist)>0 and len(brandregionlist)>0 and brandmodelist and brandmonthlist)
    
    if brandselections_changed and brandhas_valid_selections:
        
        ## Initializing state if it passes through this        
        st.session_state.selected_brand = brandlist
        st.session_state.brand_selected_retailer = brandretailerlist
        st.session_state.brand_selected_mode = brandmodelist
        st.session_state.brand_selected_month = brandmonthlist
        st.session_state.brand_selected_subcategory = brandsubcategorylist
        st.session_state.brand_selected_outlet = brandoutletlist
        st.session_state.brand_selected_region = brandregionlist
        
        # Generate and store data for graphs
        pievol = brandpievol(combineddata, brandlist, brandretailerlist, brandsubcategorylist, brandoutletlist, brandregionlist, brandmodelist, brandmonthlist)
        st.session_state.brand_matytd = pievol[0]
        st.session_state.brand_threemonth = pievol[1]
        st.session_state.brand_currentmonth = pievol[2]
        
        # Generate and store table data
        brand_three_table = brand_table_data_vol(combineddata, brandlist, brandretailerlist, brandsubcategorylist, brandoutletlist, brandregionlist, brandmodelist, brandmonthlist)
        st.session_state.brand_matytd_table = brand_three_table[0]
        st.session_state.brand_matytd_sharetable = brand_three_table[1]
        st.session_state.brand_threemonth_table = brand_three_table[2]
        st.session_state.brand_currentmonth_table = brand_three_table[3]
        
        # Mark that graph content has been generated
        st.session_state.brand_content_generated = True
        
    if st.session_state.brand_content_generated and brandhas_valid_selections:
        
        ## Pie charts    
        st.subheader("Market Share - Value",divider="gray")
        with st.container():
            matytdgraph, threemgraph, currentgraph = st.columns(3)
            with matytdgraph:
                reqmatytddf =  st.session_state.brand_matytd
                
                # Plot Details
                color_map = {
                    'MC CAIN': '#4e79a7',   # blue
                    'AL AIN': '#e15759',    # red
                    'AL KABEER': '#76b7b2', # teal
                    'SADIA': '#59a14f',     # green
                    'AL ISLAMI': '#ffcc99', # orange
                    'LAMB WESTON': 'b8f2b8', # mint
                    'SEARA' : '#ffdb4d',     # yellow
                    'Others': '#d98cb3',     # mauve
                    'AL AREESH': '#ff9999',  # light red
                    'ARDO': '#99ff99',       # light green
                    'AUNT BESSIE': '#ff6666',# light orange
                    'BIRDS EYE': '#c2c2f0',  # pale purple
                    'CO OP': '#ffb366',      # peach
                    'CO OP ISLAMI': '#99e6e6',# teal
                    'CRF': '#c2d6d6',        # grey-blue
                    'DELIGHT': '#ff8080',    # salmon
                    'EMBORG': '#b3b3ff',     # periwinkle
                    'GREEN GIANT': '#e6ac00',# mustart
                    'JEKOR':'#f2d9e6',       # pale pink
                    'JOYS': '#8cd9b3',       # seafoam
                    'LE DUC': '#d6f5d6',     # very light green
                    'MEGA': '#80bfff',     # sky blue
                    'ORE IDA': '#ffbf80',    # light coral
                    'PLEIN SOLEIL': '#9370DB', # medium purple
                    'SALUD': '#20B2AA',      # light sea green
                    'SUNBULLAH': '#CD5C5C',  # Indian Red
                    'SUPER FRESH': '#DAA520'# Goldenrod
                }
                
                fig = px.pie(reqmatytddf, names='Brand Name', values='Share', title='MAT/YTD', 
                             color_discrete_map=color_map)
                fig.update_traces(hovertemplate='<b>%{label}</b>Percentage: %{customdata:.2f}%<extra></extra>',customdata=reqmatytddf['Share'] * 100)
                st.plotly_chart(fig)
            with threemgraph:
                reqthreemgraph =  st.session_state.brand_threemonth
                fig = px.pie(reqthreemgraph, names='Brand Name', values='Share', title='Latest 3M', 
                             color_discrete_map=color_map)
                fig.update_traces(hovertemplate='<b>%{label}</b>Percentage: %{customdata:.2f}%<extra></extra>',customdata=reqthreemgraph['Share'] * 100)  # Format hover text with 2 decimals
                st.plotly_chart(fig)
            with currentgraph:
                reqcurrentgraph =  st.session_state.brand_currentmonth
                fig = px.pie(reqcurrentgraph, names='Brand Name', values='Share', title='Latest Month', 
                             color_discrete_map=color_map)
                fig.update_traces(hovertemplate='<b>%{label}</b>Percentage: %{customdata:.2f}%<extra></extra>',customdata=reqcurrentgraph['Share'] * 100)  # Format hover text with 2 decimals
                st.plotly_chart(fig)
        ## Tabular Analysis
        st.subheader("Market Share - Volume",divider="gray")
        with st.container(border=True):
            matytdtable, matytdsharetable, threemtable, currenttable = st.columns(4)
            with matytdtable:
                reqmatytdtable = st.session_state.brand_matytd_table
                reqmatytdtable['Year Ago'] = round(reqmatytdtable['Year Ago']/10**3,0)
                reqmatytdtable['This Year'] = round(reqmatytdtable['This Year']/10**3,0)
                reqmatytdtable = reqmatytdtable.rename(columns={'Year Ago':'YA (K)','This Year':'TY (K)','% Change':'%Chg'})
                st.write('Volume')
                st.dataframe(reqmatytdtable)
            with matytdsharetable:
                reqmatytdsharetable = st.session_state.brand_matytd_sharetable
                reqmatytdsharetable = reqmatytdsharetable.rename(columns={'% YA Share':'%YA','% TY Share':'%TY','% Change':'%Chg'})
                st.write('MAT/YTD')
                st.data_editor(reqmatytdsharetable,hide_index=True)
            with threemtable:
                reqthreemtable = st.session_state.brand_threemonth_table
                reqthreemtable = reqthreemtable.rename(columns={'% Current':'Latest','% Previous':'Previous','% Change':'%Chg'})
                reqthreemtable = reqthreemtable[['Previous','Latest','%Chg']]
                st.write('3M')
                st.data_editor(reqthreemtable,hide_index=True)
            with currenttable:
                reqcurrenttable = st.session_state.brand_currentmonth_table
                reqcurrenttable = reqcurrenttable.rename(columns={'% Current':'Latest','% Previous':'Previous','% Change':'%Chg'})
                reqcurrenttable = reqcurrenttable[['Previous','Latest','%Chg']]
                st.write('1M')
                st.data_editor(reqcurrenttable,hide_index=True)