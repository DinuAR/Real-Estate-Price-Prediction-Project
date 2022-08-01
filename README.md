# Real-Estate-Price-Prediction-Project
This is a Machine Learning project to predict real estate prices

![Screen Capture 002 - Bengaluru House Price Prediction - Jupyter Notebook - localhost](https://user-images.githubusercontent.com/87066711/182025875-fa35d71f-4c8c-4d8d-a3ef-d00f81927e19.jpg)

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4777d3bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "%matplotlib inline\n",
    "import matplotlib\n",
    "matplotlib.rcParams[\"figure.figsize\"] = (20,10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b79f0580",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>area_type</th>\n",
       "      <th>availability</th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>society</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>balcony</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>19-Dec</td>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>Coomee</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>39.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Plot  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>Theanmp</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>120.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>62.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>Soiewre</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>95.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>51.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              area_type   availability                  location       size  \\\n",
       "0  Super built-up  Area         19-Dec  Electronic City Phase II      2 BHK   \n",
       "1            Plot  Area  Ready To Move          Chikka Tirupathi  4 Bedroom   \n",
       "2        Built-up  Area  Ready To Move               Uttarahalli      3 BHK   \n",
       "3  Super built-up  Area  Ready To Move        Lingadheeranahalli      3 BHK   \n",
       "4  Super built-up  Area  Ready To Move                  Kothanur      2 BHK   \n",
       "\n",
       "   society total_sqft  bath  balcony   price  \n",
       "0  Coomee        1056   2.0      1.0   39.07  \n",
       "1  Theanmp       2600   5.0      3.0  120.00  \n",
       "2      NaN       1440   2.0      3.0   62.00  \n",
       "3  Soiewre       1521   3.0      1.0   95.00  \n",
       "4      NaN       1200   2.0      1.0   51.00  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = pd.read_csv(\"Bengaluru_House_Data.csv\")\n",
    "df1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a739c825",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13320, 9)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fc96f3d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "area_type\n",
       "Built-up  Area          2418\n",
       "Carpet  Area              87\n",
       "Plot  Area              2025\n",
       "Super built-up  Area    8790\n",
       "Name: area_type, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.groupby('area_type')['area_type'].agg('count')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc831310",
   "metadata": {},
   "source": [
    "## Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1c378779",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size total_sqft  bath   price\n",
       "0  Electronic City Phase II      2 BHK       1056   2.0   39.07\n",
       "1          Chikka Tirupathi  4 Bedroom       2600   5.0  120.00\n",
       "2               Uttarahalli      3 BHK       1440   2.0   62.00\n",
       "3        Lingadheeranahalli      3 BHK       1521   3.0   95.00\n",
       "4                  Kothanur      2 BHK       1200   2.0   51.00"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')\n",
    "df2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8ca043ef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location       1\n",
       "size          16\n",
       "total_sqft     0\n",
       "bath          73\n",
       "price          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "56f7d76b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      0\n",
       "size          0\n",
       "total_sqft    0\n",
       "bath          0\n",
       "price         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3 = df2.dropna()\n",
    "df3.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5026a562",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13246, 5)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a01de0d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['2 BHK', '4 Bedroom', '3 BHK', '4 BHK', '6 Bedroom', '3 Bedroom',\n",
       "       '1 BHK', '1 RK', '1 Bedroom', '8 Bedroom', '2 Bedroom',\n",
       "       '7 Bedroom', '5 BHK', '7 BHK', '6 BHK', '5 Bedroom', '11 BHK',\n",
       "       '9 BHK', '9 Bedroom', '27 BHK', '10 Bedroom', '11 Bedroom',\n",
       "       '10 BHK', '19 BHK', '16 BHK', '43 Bedroom', '14 BHK', '8 BHK',\n",
       "       '12 Bedroom', '13 BHK', '18 Bedroom'], dtype=object)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3['size'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "87f7d72e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/gw/f_j87lmj7vd4jdrc3kfcbs1h0000gn/T/ipykernel_4742/2222900254.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))\n"
     ]
    }
   ],
   "source": [
    "df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "afd80c4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size total_sqft  bath   price  bhk\n",
       "0  Electronic City Phase II      2 BHK       1056   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom       2600   5.0  120.00    4\n",
       "2               Uttarahalli      3 BHK       1440   2.0   62.00    3\n",
       "3        Lingadheeranahalli      3 BHK       1521   3.0   95.00    3\n",
       "4                  Kothanur      2 BHK       1200   2.0   51.00    2"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a1c79b2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  4,  3,  6,  1,  8,  7,  5, 11,  9, 27, 10, 19, 16, 43, 14, 12,\n",
       "       13, 18])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3['bhk'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "12ff8709",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1718</th>\n",
       "      <td>2Electronic City Phase II</td>\n",
       "      <td>27 BHK</td>\n",
       "      <td>8000</td>\n",
       "      <td>27.0</td>\n",
       "      <td>230.0</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4684</th>\n",
       "      <td>Munnekollal</td>\n",
       "      <td>43 Bedroom</td>\n",
       "      <td>2400</td>\n",
       "      <td>40.0</td>\n",
       "      <td>660.0</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       location        size total_sqft  bath  price  bhk\n",
       "1718  2Electronic City Phase II      27 BHK       8000  27.0  230.0   27\n",
       "4684                Munnekollal  43 Bedroom       2400  40.0  660.0   43"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3[df3.bhk>20]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6d46bb50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['1056', '2600', '1440', ..., '1133 - 1384', '774', '4689'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.total_sqft.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f639bd59",
   "metadata": {},
   "outputs": [],
   "source": [
    "def is_float(x):\n",
    "    try:\n",
    "        float(x)\n",
    "    except:\n",
    "        return False\n",
    "    return True    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4347a77c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>Yelahanka</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>2100 - 2850</td>\n",
       "      <td>4.0</td>\n",
       "      <td>186.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>Hebbal</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>3067 - 8156</td>\n",
       "      <td>4.0</td>\n",
       "      <td>477.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>137</th>\n",
       "      <td>8th Phase JP Nagar</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1042 - 1105</td>\n",
       "      <td>2.0</td>\n",
       "      <td>54.005</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>165</th>\n",
       "      <td>Sarjapur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1145 - 1340</td>\n",
       "      <td>2.0</td>\n",
       "      <td>43.490</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>188</th>\n",
       "      <td>KR Puram</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1015 - 1540</td>\n",
       "      <td>2.0</td>\n",
       "      <td>56.800</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               location   size   total_sqft  bath    price  bhk\n",
       "30            Yelahanka  4 BHK  2100 - 2850   4.0  186.000    4\n",
       "122              Hebbal  4 BHK  3067 - 8156   4.0  477.000    4\n",
       "137  8th Phase JP Nagar  2 BHK  1042 - 1105   2.0   54.005    2\n",
       "165            Sarjapur  2 BHK  1145 - 1340   2.0   43.490    2\n",
       "188            KR Puram  2 BHK  1015 - 1540   2.0   56.800    2"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3[~df3['total_sqft'].apply(is_float)].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "082881dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert_sqft_to_num(x):\n",
    "    tokens = x.split('-')\n",
    "    if len(tokens) == 2:\n",
    "        return (float(tokens[0])+float(tokens[1]))/2\n",
    "    try:\n",
    "        return float(x)\n",
    "    except:\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "76cdbcc3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2166.0"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "convert_sqft_to_num('2166')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "7f200353",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5611.5"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "convert_sqft_to_num('3067 - 8156')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8333b37a",
   "metadata": {},
   "outputs": [],
   "source": [
    "convert_sqft_to_num('34.46Sq. Meter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "227461ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "df4 = df3.copy()\n",
    "df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e2401c3a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4\n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "53e205c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      Yelahanka\n",
       "size              4 BHK\n",
       "total_sqft       2475.0\n",
       "bath                4.0\n",
       "price             186.0\n",
       "bhk                   4\n",
       "Name: 30, dtype: object"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4.loc[30]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8549a62",
   "metadata": {},
   "source": [
    "## Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "83100043",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4\n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "18cc9e6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "      <td>3699.810606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "      <td>4615.384615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "      <td>4305.555556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "      <td>6245.890861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "      <td>4250.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk  \\\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2   \n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4   \n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3   \n",
       "3        Lingadheeranahalli      3 BHK      1521.0   3.0   95.00    3   \n",
       "4                  Kothanur      2 BHK      1200.0   2.0   51.00    2   \n",
       "\n",
       "   price_per_sqft  \n",
       "0     3699.810606  \n",
       "1     4615.384615  \n",
       "2     4305.555556  \n",
       "3     6245.890861  \n",
       "4     4250.000000  "
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5 = df4.copy()\n",
    "df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']\n",
    "df5.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5360438c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1304"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df5.location.unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "e543bf28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location\n",
       "Whitefield               535\n",
       "Sarjapur  Road           392\n",
       "Electronic City          304\n",
       "Kanakpura Road           266\n",
       "Thanisandra              236\n",
       "                        ... \n",
       "1 Giri Nagar               1\n",
       "Kanakapura Road,           1\n",
       "Kanakapura main  Road      1\n",
       "Karnataka Shabarimala      1\n",
       "whitefiled                 1\n",
       "Name: location, Length: 1293, dtype: int64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.location = df5.location.apply(lambda x: x.strip())\n",
    "\n",
    "location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)\n",
    "location_stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ab1013c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1052"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(location_stats[location_stats<=10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "6a17f202",
   "metadata": {},
   "outputs": [],
   "source": [
    "location_stats_less_than_10 = location_stats[location_stats<=10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "288dbf5f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location\n",
       "Basapura                 10\n",
       "1st Block Koramangala    10\n",
       "Gunjur Palya             10\n",
       "Kalkere                  10\n",
       "Sector 1 HSR Layout      10\n",
       "                         ..\n",
       "1 Giri Nagar              1\n",
       "Kanakapura Road,          1\n",
       "Kanakapura main  Road     1\n",
       "Karnataka Shabarimala     1\n",
       "whitefiled                1\n",
       "Name: location, Length: 1052, dtype: int64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "location_stats_less_than_10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "ec039260",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1293"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df5.location.unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "7f37a243",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "242"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)\n",
    "len(df5.location.unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "df1106cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "      <td>3699.810606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "      <td>4615.384615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "      <td>4305.555556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "      <td>6245.890861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "      <td>4250.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Whitefield</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1170.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>38.00</td>\n",
       "      <td>2</td>\n",
       "      <td>3247.863248</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Old Airport Road</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>2732.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>204.00</td>\n",
       "      <td>4</td>\n",
       "      <td>7467.057101</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Rajaji Nagar</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>3300.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>600.00</td>\n",
       "      <td>4</td>\n",
       "      <td>18181.818182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Marathahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1310.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>63.25</td>\n",
       "      <td>3</td>\n",
       "      <td>4828.244275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>other</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>370.00</td>\n",
       "      <td>6</td>\n",
       "      <td>36274.509804</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk  \\\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2   \n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4   \n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3   \n",
       "3        Lingadheeranahalli      3 BHK      1521.0   3.0   95.00    3   \n",
       "4                  Kothanur      2 BHK      1200.0   2.0   51.00    2   \n",
       "5                Whitefield      2 BHK      1170.0   2.0   38.00    2   \n",
       "6          Old Airport Road      4 BHK      2732.0   4.0  204.00    4   \n",
       "7              Rajaji Nagar      4 BHK      3300.0   4.0  600.00    4   \n",
       "8              Marathahalli      3 BHK      1310.0   3.0   63.25    3   \n",
       "9                     other  6 Bedroom      1020.0   6.0  370.00    6   \n",
       "\n",
       "   price_per_sqft  \n",
       "0     3699.810606  \n",
       "1     4615.384615  \n",
       "2     4305.555556  \n",
       "3     6245.890861  \n",
       "4     4250.000000  \n",
       "5     3247.863248  \n",
       "6     7467.057101  \n",
       "7    18181.818182  \n",
       "8     4828.244275  \n",
       "9    36274.509804  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97be5fc7",
   "metadata": {},
   "source": [
    "## Outlier Removal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "7d54725b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>other</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>370.0</td>\n",
       "      <td>6</td>\n",
       "      <td>36274.509804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>HSR Layout</td>\n",
       "      <td>8 Bedroom</td>\n",
       "      <td>600.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>200.0</td>\n",
       "      <td>8</td>\n",
       "      <td>33333.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>58</th>\n",
       "      <td>Murugeshpalya</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1407.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>6</td>\n",
       "      <td>10660.980810</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68</th>\n",
       "      <td>Devarachikkanahalli</td>\n",
       "      <td>8 Bedroom</td>\n",
       "      <td>1350.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>85.0</td>\n",
       "      <td>8</td>\n",
       "      <td>6296.296296</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>70</th>\n",
       "      <td>other</td>\n",
       "      <td>3 Bedroom</td>\n",
       "      <td>500.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>100.0</td>\n",
       "      <td>3</td>\n",
       "      <td>20000.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               location       size  total_sqft  bath  price  bhk  \\\n",
       "9                 other  6 Bedroom      1020.0   6.0  370.0    6   \n",
       "45           HSR Layout  8 Bedroom       600.0   9.0  200.0    8   \n",
       "58        Murugeshpalya  6 Bedroom      1407.0   4.0  150.0    6   \n",
       "68  Devarachikkanahalli  8 Bedroom      1350.0   7.0   85.0    8   \n",
       "70                other  3 Bedroom       500.0   3.0  100.0    3   \n",
       "\n",
       "    price_per_sqft  \n",
       "9     36274.509804  \n",
       "45    33333.333333  \n",
       "58    10660.980810  \n",
       "68     6296.296296  \n",
       "70    20000.000000  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5[df5.total_sqft/df5.bhk<300].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "c6ddd02c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13246, 7)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "98ed66b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df6 = df5[~(df5.total_sqft/df5.bhk<300)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "f5fbacac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12502, 7)"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df6.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "62905b40",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     12456.000000\n",
       "mean       6308.502826\n",
       "std        4168.127339\n",
       "min         267.829813\n",
       "25%        4210.526316\n",
       "50%        5294.117647\n",
       "75%        6916.666667\n",
       "max      176470.588235\n",
       "Name: price_per_sqft, dtype: float64"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df6.price_per_sqft.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "492dfc42",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_pps_outliers(df):\n",
    "    df_out = pd.DataFrame()\n",
    "    for key, subdf in df.groupby('location'):\n",
    "        m = np.mean(subdf.price_per_sqft)\n",
    "        st = np.std(subdf.price_per_sqft)\n",
    "        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]\n",
    "        df_out = pd.concat([df_out, reduced_df], ignore_index=True)\n",
    "    return df_out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "ac5ec880",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10241, 7)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df7 = remove_pps_outliers(df6)\n",
    "df7.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "d266a061",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3sAAAJcCAYAAABAE73ZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA5xElEQVR4nO3dfZikZ10n+u9vkiFDZlqTkGDCBDc5JCx5EUZokD3O0c4iBHLpQXlx4sXxkEP24OHlaIi6oLtHQGU3i6voIsoFyE6WRZkB2TXLAQSDMzjKy04wvCTRnTmSmCEBxphgZyTJJHOfP6o605mp7nTPdHV1P/X5XFdfVX0/T1X9qlMp+OZ3P/ddrbUAAADQLWtGXQAAAABLT9gDAADoIGEPAACgg4Q9AACADhL2AAAAOkjYAwAA6CBhDwAWqKreVFX/eZ7jt1bVDx3jc++oqn9x7NUBwCMJewCMlUGBrKquqKpdo6oJAIZB2AMAAOggYQ8AZqmqJ1TVH1bV/qr6alX99BGnrKuqbVU1XVVfqKqnHXH8mVV1c1XdXVX/sarW9Z/31Kr6SP957+7fP3t53hUA40jYA4C+qlqT5L8l+WKSjUmek+Sqqrp01mkvTPLBJKcl+f0k/7Wq1s46/rIklyZ5UpInJ/nX/fE1Sf5jkn+S5LuTfDvJbw/tzQAw9oQ9AMbRf62qe2Z+kvxOf/yZSc5orf1ya+2B1trfJHl3kstnPfaG1tqHWmsHk/xGknVJnj3r+G+31m5vrf19krck+Ykkaa3d1Vr7w9baP7bWpvvHfnCo7xKAsXbiqAsAgBH40dban8z8UlVXJPkX6XXdntAPgDNOSPJns36/feZOa+1QVe1L8oRBx5PcNnOsqk5O8rYkz09yav/4RFWd0Fp76HjfEAAcSdgDgMNuT/LV1tr585zzxJk7/WmfZye5Y9Dx9KZrzhz72ST/NMn3tda+XlWbkvxlklqCugHgKKZxAsBhn0/yD1X1+qp6bFWdUFUXV9UzZ53zjKp6UVWdmOSqJPcn+eys46+pqrOr6rQkv5hkW398Ir3r9O7pH3vj0N8NAGNN2AOAvv50yh9JsinJV5P8XZL3JPnOWaf9UZItSe5O8pNJXtS/fm/G7yf5RJK/6f/8an/8N5M8tv+cn03y8SG9DQBIklRrbdQ1AAAAsMR09gAAADpI2AMAAOggYQ8AAKCDhD0AAIAOWtX77J1++untnHPOGXUZAAAAI3HDDTf8XWvtjEHHVnXYO+ecc7J79+5RlwEAADASVXXbXMdM4wQAAOggYQ8AAKCDhD0AAIAOWtXX7A1y8ODB7Nu3L/fdd9+oSxmpdevW5eyzz87atWtHXQoAADACnQt7+/bty8TERM4555xU1ajLGYnWWu66667s27cv55577qjLAQAARqBz0zjvu+++PO5xjxvboJckVZXHPe5xY9/dBACAcda5sJdkrIPeDH8DAAAYb50MewAAAONO2Ftit99+ey655JJccMEFueiii/Jbv/VbA89705velI0bN2bTpk15ylOekle96lU5dOhQkuSKK67Ihz70oUecv2HDhiTJrbfemosvvvjh8Xe/+915+tOfnrvvvntI7wgAAFiNxj7sTU8n73lP8vrX926np4/v+U488cT8+q//em655ZZ89rOfzTve8Y7cfPPNA8993etelxtvvDE333xzvvzlL2fnzp2Leq33ve99efvb355PfOITOfXUU4+vcAAAoFM6txrnYuzalVx2WXLoUHLgQLJ+fXL11clHP5ps3nxsz3nWWWflrLPOSpJMTEzkggsuyNe+9rVceOGFcz7mgQceyH333beowLZ9+/Zcc801uf7663P66acfW7EAAEBnjW1nb3q6F/Smp3tBL+ndzozfe+/xv8att96av/zLv8z3fd/3DTz+tre9LZs2bcpZZ52VJz/5ydm0adPDx37+538+mzZtevhntttuuy2vfe1r84lPfCJnnnnm8RcKAAB0ztiGvW3beh29QQ4d6h0/Hvfee29e/OIX5zd/8zfzHd/xHQPPmZnG+c1vfjMHDhzIBz7wgYeP/dqv/VpuvPHGh39mO+OMM/Ld3/3d2b59+/EVCQAAdNbYhr09ew539I504ECyd++xP/fBgwfz4he/OC972cvyohe96FHPX7t2bZ7//Ofn05/+9IKe/+STT87HPvaxvPOd78z73//+Yy8UAADorLG9Zu/883vX6A0KfOvXJ+edd2zP21rLlVdemQsuuCBXX331gh/zF3/xF0dN15zPGWeckY9//OOZmprK6aefnksvvfTYCgYAADppbDt7W7Yka+Z492vW9I4fiz//8z/P+973vnzqU596+Hq7j370owPPnblm7+KLL86DDz6YV7/61Yt6rXPPPTfXXXddXvGKV+Rzn/vcsRUMAAB0UrXWRl3DMZucnGy7d+9+xNgtt9ySCy64YEGPH7Qa55o1x7ca50qymL8FAACw+lTVDa21yUHHxnYaZ9ILdHfc0VuMZe/e3tTNLVuS/v7lAAAAq9ZYh72kF+yuvHLUVQAAACytsb1mDwAAYCGmtk5lauvUqMtYNGEPAACgg4Q9AACADhr7a/YAAACONHva5s7bdh41tuOKHctb0DHQ2Vti9913X571rGflaU97Wi666KK88Y1vHHjem970pmzcuDGbNm3KU57ylLzqVa/KoUOHkiRXXHFFPvShDz3i/A39JUJvvfXWXHzxxQ+Pv/vd787Tn/703H333UN6RwAAwGqks5fDCX0p0vlJJ52UT33qU9mwYUMOHjyYzZs35wUveEGe/exnH3Xu6173uvzcz/1cDh06lB/4gR/Izp07c8kllyz4td73vvfl7W9/ez71qU/l1FNPPe7aAQCAntnZYCnzwnIS9pZYVT3chTt48GAOHjyYqpr3MQ888EDuu+++RQW27du355prrsn111+f008//bhqBgAAusc0ziF46KGHsmnTpjz+8Y/Pc5/73Hzf933fwPPe9ra3ZdOmTTnrrLPy5Cc/OZs2bXr42M///M9n06ZND//Mdtttt+W1r31tPvGJT+TMM88c4jsBAABWq7Ht7A3zgssTTjghN954Y+6555782I/9WL7yla884jq7GTPTOA8ePJiXvOQl+cAHPpDLL788SfJrv/ZreclLXvLwuTPdwiQ544wzctppp2X79u153eted8x1AgAAj261Td+cobM3RKecckqmpqby8Y9/fN7z1q5dm+c///n59Kc/vaDnPfnkk/Oxj30s73znO/P+979/KUoFAAA6Zmw7e8O64HL//v1Zu3ZtTjnllHz729/On/zJn+T1r3/9vI9preUv/uIvjpquOZ8zzjgjH//4xzM1NZXTTz89l1566XFWDgAAdInO3hK78847c8kll+SpT31qnvnMZ+a5z31ufviHf3jguTPX7F188cV58MEH8+pXv3pRr3Xuuefmuuuuyyte8Yp87nOfW4ryAQCAjqjW2qhrOGaTk5Nt9+7djxi75ZZbcsEFFyzqeVbrUqqP5lj+FgAAwOpRVTe01iYHHRvbaZyzdS3kAQAAmMYJAADQQZ0Me6t5aupS8TcAAIDx1rmwt27dutx1111j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\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_scatter_chart(df,location):\n",
    "    bhk2 = df[(df.location == location) &(df.bhk==2)]\n",
    "    bhk3 = df[(df.location == location) &(df.bhk==3)]\n",
    "    matplotlib.rcParams['figure.figsize'] = (15,10)\n",
    "    plt.scatter(bhk2.total_sqft, bhk2.price,color='blue',label='2 BHK', s=50)\n",
    "    plt.scatter(bhk3.total_sqft, bhk3.price,marker='+',color='green',label='3 BHK', s=50)\n",
    "    plt.xlabel(\"Total square feet area\")\n",
    "    plt.ylabel(\"Price\")\n",
    "    plt.title(location)\n",
    "    plt.legend()\n",
    "    \n",
    "plot_scatter_chart(df7,\"Hebbal\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c2f1f46",
   "metadata": {},
   "source": [
    "### We should also remove properties where for same location, he price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft.area). we will build a dictionary of stats per bhk, i.e.\n",
    "\n",
    "## {\n",
    "    '1' : {\n",
    "        'mean' : 4000,\n",
    "        'std' : 2000,\n",
    "        'count' : 34\n",
    "    },\n",
    "    '2' : {\n",
    "        'mean' : 4300,\n",
    "        'std' : 2300,\n",
    "        'count' : 22\n",
    "    },\n",
    "## }\n",
    "\n",
    "### Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1BHK apartment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "bf23f348",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_bhk_outliers(df):\n",
    "    exclude_indices = np.array([])\n",
    "    for location, location_df in df.groupby('location'):\n",
    "        bhk_stats = {}\n",
    "        for bhk, bhk_df in location_df.groupby('bhk'):\n",
    "            bhk_stats[bhk] = {\n",
    "                'mean' : np.mean(bhk_df.price_per_sqft),\n",
    "                'std' : np.std(bhk_df.price_per_sqft),\n",
    "                'count' : bhk_df.shape[0]\n",
    "            }\n",
    "        for bhk, bhk_df in location_df.groupby('bhk'):\n",
    "            stats = bhk_stats.get(bhk-1)\n",
    "            if stats and stats['count']>5:\n",
    "                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)\n",
    "    return df.drop(exclude_indices, axis='index')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "0cb6846b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7329, 7)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df8 = remove_bhk_outliers(df7)\n",
    "df8.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "6b04af7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3sAAAJcCAYAAABAE73ZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA4Y0lEQVR4nO3de5ScZ30n+O9PtmJhqRPb2MRGJrEXm8GXAQXEZTbapD2Em0+yQIC0c9gsXjwLy2UT4ySDycwm5MKMByYhDJBwgDDyMCRIIczEkwUCMZGIEi4jE3OxHUbaYMfCBhTHJm2Bbdl69o+qttpSqdUtdXV1v/X5nNOnqp/3rbd+LYpKvvye93mqtRYAAAC6ZdWoCwAAAGDxCXsAAAAdJOwBAAB0kLAHAADQQcIeAABABwl7AAAAHSTsAcA8VdWbquo/z3H81qr6sWO89raq+hfHXh0APJKwB8BYGRTIquryqtoxqpoAYBiEPQAAgA4S9gBglqp6bFX9UVXtraqvVdXPHnLKmqraUlXTVfWFqnryIcefVlU3V9XdVfUfq2pN/7qnVtWf9K97d//52UvzVwEwjoQ9AOirqlVJ/luSLyZZn+RZSa6squfOOu0FSf4wyWlJfj/Jf62q1bOOvyzJc5M8PskTkvzr/viqJP8xyQ8m+YEk303yzqH9MQCMPWEPgHH0X6vqnpmfJL/TH39akjNaa7/WWnugtfa3Sd6b5LJZr72htfbh1tr+JL+VZE2SZ846/s7W2u2ttX9I8uYkP50krbW7Wmt/1Fr7Tmttun/sR4f6VwIw1k4cdQEAMAIvbK392cwvVXV5kn+RXtftsf0AOOOEJH8x6/fbZ5601g5U1Z4kjx10PMltM8eq6uQkb0vyvCSn9o9PVNUJrbWHjvcPAoBDCXsAcNDtSb7WWjt/jnMeN/OkP+3z7CR3DDqe3nTNmWM/n+SfJHlGa+0bVbUhyV8nqUWoGwAOYxonABz0+ST/WFVvqKpHVdUJVXVxVT1t1jlPraqfrKoTk1yZ5P4kn511/LVVdXZVnZbkl5Js6Y9PpHef3j39Y78y9L8GgLEm7AFAX3865U8k2ZDka0n+Psn7knzfrNP+OMlUkruT/EySn+zfvzfj95N8Isnf9n9+oz/+20ke1b/mZ5N8fEh/BgAkSaq1NuoaAAAAWGQ6ewAAAB0k7AEAAHSQsAcAANBBwh4AAEAHreh99k4//fR2zjnnjLoMAACAkbjhhhv+vrV2xqBjKzrsnXPOOdm5c+eoywAAABiJqrrtSMdM4wQAAOggYQ8AAKCDhD0AAIAOWtH37A2yf//+7NmzJ/fdd9+oSxmpNWvW5Oyzz87q1atHXQoAADACnQt7e/bsycTERM4555xU1ajLGYnWWu66667s2bMn55577qjLAQAARqBz0zjvu+++PPrRjx7boJckVZVHP/rRY9/dBACAcda5sJdkrIPeDP8GAAAw3joZ9gAAAMadsLfIbr/99lxyySW54IILctFFF+Xtb3/7wPPe9KY3Zf369dmwYUOe+MQn5tWvfnUOHDiQJLn88svz4Q9/+BHnr1u3Lkly66235uKLL354/L3vfW+e8pSn5O677x7SXwQAAKxEYx/2pqeT970vecMbeo/T08d3vRNPPDG/+Zu/mVtuuSWf/exn8653vSs333zzwHNf//rX58Ybb8zNN9+cL3/5y9m+ffuC3usDH/hA3vGOd+QTn/hETj311OMrHAAA6JTOrca5EDt2JJdemhw4kOzbl6xdm1x1VfLRjyabNh3bNc8666ycddZZSZKJiYlccMEF+frXv54LL7zwiK954IEHct999y0osG3dujXXXHNNrr/++px++unHViwAANBZY9vZm57uBb3p6V7QS3qPM+P33nv873Hrrbfmr//6r/OMZzxj4PG3ve1t2bBhQ84666w84QlPyIYNGx4+9ou/+IvZsGHDwz+z3XbbbXnd616XT3ziEznzzDOPv1AAAKBzxjbsbdnS6+gNcuBA7/jxuPfee/PiF784v/3bv53v/d7vHXjOzDTOb33rW9m3b18+9KEPPXzsrW99a2688caHf2Y744wz8gM/8APZunXr8RUJAAB01tiGvV27Dnb0DrVvX7J797Ffe//+/Xnxi1+cl73sZfnJn/zJo56/evXqPO95z8unP/3peV3/5JNPzsc+9rG8+93vzgc/+MFjLxQAAOissb1n7/zze/foDQp8a9cm5513bNdtreWKK67IBRdckKuuumrer/mrv/qrw6ZrzuWMM87Ixz/+8UxOTub000/Pc5/73GMrGAAA6KSx7exNTSWrjvDXr1rVO34s/vIv/zIf+MAH8qlPferh++0++tGPDjx35p69iy++OA8++GBe85rXLOi9zj333Fx33XV5xSte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\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_scatter_chart(df8,\"Hebbal\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "b970d111",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Count')"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib\n",
    "matplotlib.rcParams[\"figure.figsize\"] = (20,10)\n",
    "plt.hist(df8.price_per_sqft,rwidth=0.8)\n",
    "plt.xlabel(\"Price per squrefeet\")\n",
    "plt.ylabel(\"Count\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "fbb421cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4.,  3.,  2.,  5.,  8.,  1.,  6.,  7.,  9., 12., 16., 13.])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df8.bath.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "16e1da3c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5277</th>\n",
       "      <td>Neeladri Nagar</td>\n",
       "      <td>10 BHK</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>10</td>\n",
       "      <td>4000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8486</th>\n",
       "      <td>other</td>\n",
       "      <td>10 BHK</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>525.0</td>\n",
       "      <td>10</td>\n",
       "      <td>4375.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8575</th>\n",
       "      <td>other</td>\n",
       "      <td>16 BHK</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>550.0</td>\n",
       "      <td>16</td>\n",
       "      <td>5500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9308</th>\n",
       "      <td>other</td>\n",
       "      <td>11 BHK</td>\n",
       "      <td>6000.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>11</td>\n",
       "      <td>2500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9639</th>\n",
       "      <td>other</td>\n",
       "      <td>13 BHK</td>\n",
       "      <td>5425.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>275.0</td>\n",
       "      <td>13</td>\n",
       "      <td>5069.124424</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            location    size  total_sqft  bath  price  bhk  price_per_sqft\n",
       "5277  Neeladri Nagar  10 BHK      4000.0  12.0  160.0   10     4000.000000\n",
       "8486           other  10 BHK     12000.0  12.0  525.0   10     4375.000000\n",
       "8575           other  16 BHK     10000.0  16.0  550.0   16     5500.000000\n",
       "9308           other  11 BHK      6000.0  12.0  150.0   11     2500.000000\n",
       "9639           other  13 BHK      5425.0  13.0  275.0   13     5069.124424"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df8[df8.bath>10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "2672e079",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Count')"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(df8.bath,rwidth=0.8)\n",
    "plt.xlabel(\"Number of bathrooms\")\n",
    "plt.ylabel(\"Count\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "a03fc0e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1626</th>\n",
       "      <td>Chikkabanavar</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2460.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>4</td>\n",
       "      <td>3252.032520</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5238</th>\n",
       "      <td>Nagasandra</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>7000.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>450.0</td>\n",
       "      <td>4</td>\n",
       "      <td>6428.571429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6711</th>\n",
       "      <td>Thanisandra</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1806.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>116.0</td>\n",
       "      <td>3</td>\n",
       "      <td>6423.034330</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8411</th>\n",
       "      <td>other</td>\n",
       "      <td>6 BHK</td>\n",
       "      <td>11338.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>6</td>\n",
       "      <td>8819.897689</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           location       size  total_sqft  bath   price  bhk  price_per_sqft\n",
       "1626  Chikkabanavar  4 Bedroom      2460.0   7.0    80.0    4     3252.032520\n",
       "5238     Nagasandra  4 Bedroom      7000.0   8.0   450.0    4     6428.571429\n",
       "6711    Thanisandra      3 BHK      1806.0   6.0   116.0    3     6423.034330\n",
       "8411          other      6 BHK     11338.0   9.0  1000.0    6     8819.897689"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df8[df8.bath>df8.bhk+2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "211b89aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7251, 7)"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df9 = df8[df8.bath<df8.bhk+2]\n",
    "df9.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "777a6bde",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              location  total_sqft  bath  price  bhk\n",
       "0  1st Block Jayanagar      2850.0   4.0  428.0    4\n",
       "1  1st Block Jayanagar      1630.0   3.0  194.0    3\n",
       "2  1st Block Jayanagar      1875.0   2.0  235.0    3"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')\n",
    "df10.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8751c804",
   "metadata": {},
   "source": [
    "## Model Building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "a5c8e0bc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>7th Phase JP Nagar</th>\n",
       "      <th>8th Phase JP Nagar</th>\n",
       "      <th>9th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "      <th>other</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 242 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   1st Block Jayanagar  1st Phase JP Nagar  2nd Phase Judicial Layout  \\\n",
       "0                    1                   0                          0   \n",
       "1                    1                   0                          0   \n",
       "2                    1                   0                          0   \n",
       "3                    1                   0                          0   \n",
       "4                    1                   0                          0   \n",
       "\n",
       "   2nd Stage Nagarbhavi  5th Block Hbr Layout  5th Phase JP Nagar  \\\n",
       "0                     0                     0                   0   \n",
       "1                     0                     0                   0   \n",
       "2                     0                     0                   0   \n",
       "3                     0                     0                   0   \n",
       "4                     0                     0                   0   \n",
       "\n",
       "   6th Phase JP Nagar  7th Phase JP Nagar  8th Phase JP Nagar  \\\n",
       "0                   0                   0                   0   \n",
       "1                   0                   0                   0   \n",
       "2                   0                   0                   0   \n",
       "3                   0                   0                   0   \n",
       "4                   0                   0                   0   \n",
       "\n",
       "   9th Phase JP Nagar  ...  Vishveshwarya Layout  Vishwapriya Layout  \\\n",
       "0                   0  ...                     0                   0   \n",
       "1                   0  ...                     0                   0   \n",
       "2                   0  ...                     0                   0   \n",
       "3                   0  ...                     0                   0   \n",
       "4                   0  ...                     0                   0   \n",
       "\n",
       "   Vittasandra  Whitefield  Yelachenahalli  Yelahanka  Yelahanka New Town  \\\n",
       "0            0           0               0          0                   0   \n",
       "1            0           0               0          0                   0   \n",
       "2            0           0               0          0                   0   \n",
       "3            0           0               0          0                   0   \n",
       "4            0           0               0          0                   0   \n",
       "\n",
       "   Yelenahalli  Yeshwanthpur  other  \n",
       "0            0             0      0  \n",
       "1            0             0      0  \n",
       "2            0             0      0  \n",
       "3            0             0      0  \n",
       "4            0             0      0  \n",
       "\n",
       "[5 rows x 242 columns]"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dummies = pd.get_dummies(df10.location)\n",
    "dummies.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "b923b003",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 246 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "              location  total_sqft  bath  price  bhk  1st Block Jayanagar  \\\n",
       "0  1st Block Jayanagar      2850.0   4.0  428.0    4                    1   \n",
       "1  1st Block Jayanagar      1630.0   3.0  194.0    3                    1   \n",
       "2  1st Block Jayanagar      1875.0   2.0  235.0    3                    1   \n",
       "3  1st Block Jayanagar      1200.0   2.0  130.0    3                    1   \n",
       "4  1st Block Jayanagar      1235.0   2.0  148.0    2                    1   \n",
       "\n",
       "   1st Phase JP Nagar  2nd Phase Judicial Layout  2nd Stage Nagarbhavi  \\\n",
       "0                   0                          0                     0   \n",
       "1                   0                          0                     0   \n",
       "2                   0                          0                     0   \n",
       "3                   0                          0                     0   \n",
       "4                   0                          0                     0   \n",
       "\n",
       "   5th Block Hbr Layout  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                     0  ...            0                     0   \n",
       "1                     0  ...            0                     0   \n",
       "2                     0  ...            0                     0   \n",
       "3                     0  ...            0                     0   \n",
       "4                     0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "2                   0            0           0               0          0   \n",
       "3                   0            0           0               0          0   \n",
       "4                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "2                   0            0             0  \n",
       "3                   0            0             0  \n",
       "4                   0            0             0  \n",
       "\n",
       "[5 rows x 246 columns]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df11 = pd.concat([df10,dummies.drop('other',axis='columns')], axis='columns')\n",
    "df11.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "a25640df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 245 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   total_sqft  bath  price  bhk  1st Block Jayanagar  1st Phase JP Nagar  \\\n",
       "0      2850.0   4.0  428.0    4                    1                   0   \n",
       "1      1630.0   3.0  194.0    3                    1                   0   \n",
       "\n",
       "   2nd Phase Judicial Layout  2nd Stage Nagarbhavi  5th Block Hbr Layout  \\\n",
       "0                          0                     0                     0   \n",
       "1                          0                     0                     0   \n",
       "\n",
       "   5th Phase JP Nagar  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                   0  ...            0                     0   \n",
       "1                   0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "\n",
       "[2 rows x 245 columns]"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df12 = df11.drop('location', axis='columns')\n",
    "df12.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "ea15d4eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7251, 245)"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df12.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "8e03a13d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df12.drop('price', axis='columns')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "590928b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>bhk</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 244 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   total_sqft  bath  bhk  1st Block Jayanagar  1st Phase JP Nagar  \\\n",
       "0      2850.0   4.0    4                    1                   0   \n",
       "1      1630.0   3.0    3                    1                   0   \n",
       "2      1875.0   2.0    3                    1                   0   \n",
       "3      1200.0   2.0    3                    1                   0   \n",
       "4      1235.0   2.0    2                    1                   0   \n",
       "\n",
       "   2nd Phase Judicial Layout  2nd Stage Nagarbhavi  5th Block Hbr Layout  \\\n",
       "0                          0                     0                     0   \n",
       "1                          0                     0                     0   \n",
       "2                          0                     0                     0   \n",
       "3                          0                     0                     0   \n",
       "4                          0                     0                     0   \n",
       "\n",
       "   5th Phase JP Nagar  6th Phase JP Nagar  ...  Vijayanagar  \\\n",
       "0                   0                   0  ...            0   \n",
       "1                   0                   0  ...            0   \n",
       "2                   0                   0  ...            0   \n",
       "3                   0                   0  ...            0   \n",
       "4                   0                   0  ...            0   \n",
       "\n",
       "   Vishveshwarya Layout  Vishwapriya Layout  Vittasandra  Whitefield  \\\n",
       "0                     0                   0            0           0   \n",
       "1                     0                   0            0           0   \n",
       "2                     0                   0            0           0   \n",
       "3                     0                   0            0           0   \n",
       "4                     0                   0            0           0   \n",
       "\n",
       "   Yelachenahalli  Yelahanka  Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0               0          0                   0            0             0  \n",
       "1               0          0                   0            0             0  \n",
       "2               0          0                   0            0             0  \n",
       "3               0          0                   0            0             0  \n",
       "4               0          0                   0            0             0  \n",
       "\n",
       "[5 rows x 244 columns]"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "26009667",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    428.0\n",
       "1    194.0\n",
       "2    235.0\n",
       "3    130.0\n",
       "4    148.0\n",
       "Name: price, dtype: float64"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df12.price\n",
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "07f996f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "55370d79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8452277697874336"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "lr_clf = LinearRegression()\n",
    "lr_clf.fit(X_train,y_train)\n",
    "lr_clf.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "a77dae96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.82430186, 0.77166234, 0.85089567, 0.80837764, 0.83653286])"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import ShuffleSplit\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)\n",
    "\n",
    "cross_val_score(LinearRegression(), X, y, cv=cv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "72abca8e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/tree/_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>model</th>\n",
       "      <th>best_score</th>\n",
       "      <th>best_params</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>linear_regression</td>\n",
       "      <td>0.818354</td>\n",
       "      <td>{'normalize': True}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>lasso</td>\n",
       "      <td>0.687429</td>\n",
       "      <td>{'alpha': 1, 'selection': 'cyclic'}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>decision_tree</td>\n",
       "      <td>0.716943</td>\n",
       "      <td>{'criterion': 'mse', 'splitter': 'best'}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               model  best_score                               best_params\n",
       "0  linear_regression    0.818354                       {'normalize': True}\n",
       "1              lasso    0.687429       {'alpha': 1, 'selection': 'cyclic'}\n",
       "2      decision_tree    0.716943  {'criterion': 'mse', 'splitter': 'best'}"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "\n",
    "def find_best_model_using_gridsearchcv(X,y):\n",
    "    algos = {\n",
    "        'linear_regression' : {\n",
    "            'model' : LinearRegression(),\n",
    "            'params' : {\n",
    "                'normalize' : [True,False]\n",
    "            }\n",
    "        },\n",
    "        'lasso' : {\n",
    "            'model' : Lasso(),\n",
    "            'params' : {\n",
    "                'alpha' : [1,2],\n",
    "                'selection' : ['random', 'cyclic']\n",
    "            }\n",
    "        },\n",
    "        'decision_tree' : {\n",
    "            'model' : DecisionTreeRegressor(),\n",
    "            'params' : {\n",
    "                'criterion' : ['mse', 'friedman_mse'],\n",
    "                'splitter' : ['best','random']\n",
    "            }\n",
    "        }\n",
    "    }\n",
    "    scores = []\n",
    "    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)\n",
    "    for algo_name, config in algos.items():\n",
    "        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)\n",
    "        gs.fit(X,y)\n",
    "        scores.append({\n",
    "            'model' : algo_name,\n",
    "            'best_score' : gs.best_score_,\n",
    "            'best_params' : gs.best_params_\n",
    "        })\n",
    "        \n",
    "    return pd.DataFrame(scores,columns=['model','best_score','best_params'])\n",
    "\n",
    "find_best_model_using_gridsearchcv(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "1ed39950",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_price(location, sqft, bath, bhk):\n",
    "    loc_index = np.where(X.columns==location)[0][0]\n",
    "    \n",
    "    x = np.zeros(len(X.columns))\n",
    "    x[0] = sqft\n",
    "    x[1] = bath\n",
    "    x[2] = bhk\n",
    "    if loc_index >= 0:\n",
    "        x[loc_index] = 1\n",
    "        \n",
    "    return lr_clf.predict([x])[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "34664b5f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "83.49904677189352"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predict_price('1st Phase JP Nagar', 1000, 2, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "0a985436",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "86.80519395216025"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predict_price('1st Phase JP Nagar', 1000, 3, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "8cbf4c90",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/dinushiariyasena/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "181.27815484006703"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predict_price('Indira Nagar', 1000, 2, 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee4aee70",
   "metadata": {},
   "source": [
    "## Export model to pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "be79248f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('bangore_home_prices_model.pickle', 'wb') as f:\n",
    "    pickle.dump(lr_clf,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "e1b98477",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "columns = {\n",
    "    'data_columns' : [col.lower() for col in X.columns]\n",
    "}\n",
    "with open(\"columns.json\", \"w\") as f:\n",
    "    f.write(json.dumps(columns))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a247ba2",
   "metadata": {},
   "source": [
    "## Python Flask Server"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3f82919",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
