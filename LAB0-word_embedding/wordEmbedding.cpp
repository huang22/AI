#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<string>
#include<cstdio>
#include<vector>
#include<cmath>
#include<string>
#include<cstring>

using namespace std;

vector<string> file[5001];
vector<string> c;
vector<int> one_hot_[5001];
vector<double> TF_[5001];
vector<int> smat[5001];
int file_num = 0;
int word_num = 0;
int word_exit_in_file[5001] = {0};

/**************************
********ʹ�ù�ϣ��***********
*****************************

#define HashTableSize 10000

string s[HashTableSize],t[HashTableSize]; 
int find_sum = 0;//���Ҵ��� 

struct Node 
{
	char * str;
	Node * next;
};

unsigned int ELFHash(const char *str)
{
	unsigned int hash = 0;
	unsigned int x = 0;

	while (*str)
	{
		hash = (hash << 4) + (*str++); 
		if ((x = hash & 0xF0000000L) != 0)
		{
			hash ^= (x >> 24);
			hash &= ~x;
		}
	}
	return (hash & 0x7FFFFFFF);
}

void HashInsert(char * str,Node ** HashTable)
{
	int key;
	Node * tempA, * tempB;
	//Ԫ�صĹ�ϣֵ 
	 key = ELFHash(str) % (HashTableSize - 10);

	//������ֵû�б�ռ�ã���ֱ��ӳ�䵽���ֵ�� 
	if (HashTable[key] == NULL) 
	{
		HashTable[key] = (Node *)malloc(sizeof(Node));
		HashTable[key]->str = str;
		HashTable[key]->next = NULL;
		return;
	}

	tempA = HashTable[key];
	//������ֵ��ռ�ã��Ǿ�����������Ҳ������ԭ��ֵ��ָ��ָ���µ�ֵ 
	while (tempA != NULL) 
	{
		if (!strcmp(tempA->str, str)) 
			return;
		tempB = tempA;
		tempA = tempA->next;
    }

    tempA = (Node *)malloc(sizeof(Node));

    tempA->str = str;
	tempA->next = NULL;
	tempB->next = tempA;
	return;
}

//���Һ��� 
bool HashSearch(const char * str, Node ** HashTable)
{
	//����ĴʵĹ�ϣֵ���������� 
	int key = ELFHash(str) % (HashTableSize - 10);
	Node * temp;

	//��Ϊ�ջ������ֵû�б�ռ�ã�����˵������ʲ��ڵ��ʱ��� 
	if (HashTable == NULL || HashTable[key] == NULL) 
	{
		find_sum++;//��û���ҵ�������һ�� 
		return false;
	}
		
	temp = HashTable[key];
	//����ҵ��ͷ���ָ�룬��Ȼ��һֱ�ҵ�ֵû�б�ռ����֤�����ʲ��ڱ��� 
	while (temp != NULL) 
	{
		find_sum++;
		if (!strcmp(temp->str, str))
		{
			return true;	
		} 

		temp = temp->next;
	}
	return false;
}
//��ʼ��һ����ϣ�� 
Node ** HashInit(int size)
{
	Node ** HashTable = (Node **)calloc(size, sizeof(Node *));
	for(int i = 0 ; i < HashTableSize; i++)
	{
		HashTable[i] = NULL;
	}
	return HashTable;
}

//����һ���ʻ��	
void vocabulary() 
{
	vector<string> s;
	string line;						
	int n = 0,flag;
	
	ifstream in("semeval");

	//����û���������ļ����ͱ��� 
	if (!in.is_open())		
	{
		cout<<"Error opening file";
		exit (1);
	}
	
	//file_num���������ļ���Ŀ 
	file_num = 0;
	Node ** HashTable = NULL;
    HashTable = HashInit(HashTableSize);
			
	while(!in.eof())	
	{
		string str;
		//���ļ���ȡ��һ�� 
		getline(in,line);
		
		//�����������п��У�������� 
		if(line == "")
		{
			continue;
		}
		
		file_num++;
		
		//�ӵڶ���tab֮����Ǵʻ��� 
		int tab = line.find('\t',10);	
		//�Ѵʻ��һ�����з��� 
		str = line.substr(tab + 1,line.length() - tab - 1)+" ";	
		
		while(str.length() != 0)
		{
			int blank = str.find(" ");
			file[file_num].push_back(str.substr(0,blank));	
			s.push_back(str.substr(0,blank));			
			str = str.erase(0,blank + 1);
		}			 
	}
	in.close();

	//�Ž�c���vector ���� 
	for(int i = 0; i < s.size(); i++) 
	{
		char *tmp = (char*)s[i].data();
		if(!HashSearch(tmp, HashTable))
		{
			c.push_back(s[i]);
			HashInsert(tmp,HashTable);
		}
	}	
	cout<<"Search "<<find_sum<<" times."<<endl;
}
*****************/ 

/**************************
********δʹ�ù�ϣ��***********
*****************************/
//����һ���ʻ��	
void vocabulary() 
{
	vector<string> s;
	string line;						
	int n = 0,flag;
	
	ifstream in("semeval");

	//����û���������ļ����ͱ��� 
	if (!in.is_open())		
	{
		cout<<"Error opening file";
		exit (1);
	}
	
	//file_num���������ļ���Ŀ 
	file_num = 0;
			
	while(!in.eof())	
	{
		string str;
		//���ļ���ȡ��һ�� 
		getline(in,line);
		
		//�����������п��У�������� 
		if(line == "")
		{
			continue;
		}
		
		file_num++;
		
		//�ӵڶ���tab֮����Ǵʻ��� 
		int tab = line.find('\t',10);	
		//�Ѵʻ��һ�����з��� 
		str = line.substr(tab + 1,line.length() - tab - 1)+" ";	
		
		while(str.length() != 0)
		{
			int blank = str.find(" ");
			file[file_num].push_back(str.substr(0,blank));	
			s.push_back(str.substr(0,blank));
			str = str.erase(0,blank + 1);
		}			 
	}
	in.close();

	int find_sum = 0;
	
	//�õ��ʻ��c 
	c.push_back(s[0]);
	for(int i = 1; i < s.size(); i++) 
	{
		flag = 0;
		for(int j = 0; j < i; j++)
		{
			find_sum++;
			if(s[i] == s[j])		
			{
				flag = 1;
				break;
			}
		}
		if(flag != 1 && s[i] != "\n")
		{
			c.push_back(s[i]);
		}
	}	
	cout<<"Search "<<find_sum<<" times."<<endl;

}
/**************/

//���һ���ļ����Ƿ���ĳ���ʻ� 
int in_file(string str,int num)
{
	int word_exit_sum = 0;
	for(int i = 0; i < file[num].size(); i++)
	{
		if(str == file[num][i])
		{
			word_exit_sum++;
		}
	}
	return word_exit_sum;
}

//one_hot���� 
void one_hot_func()
{
	ofstream one_hot("onehot.txt"); 
	
	//����ĳ���ʻ���֣������1���������0 
	//�Ҽ�¼����ʻ��ڸ��ļ��г��ִ���������TF������� 
	for(int i = 1; i <= file_num; i++)
	{
		for(int j = 0; j < c.size(); j++)
		{
			if(in_file(c[j],i) != 0)
			{
				one_hot<<"1"<<" ";
				word_num++;
				one_hot_[i].push_back(in_file(c[j],i));
			}
			else
			{
				one_hot<<"0"<<" ";
				one_hot_[i].push_back(0);
			}
		}
		one_hot<<endl;
	}
	one_hot.close();
}

//����TF���� 
void TF_func()
{
	ofstream TF("TF.txt"); 
	//���TF����֮��ͬʱ�Ž�TF_���vector��¼�������������TF_TDF���� 
	for(int i = 1 ; i <= file_num; i++)
	{
		double word_sum = file[i].size();

		for(int j = 0; j < c.size(); j++)
		{
			if(word_sum != 0)
			{
			
				TF_[i].push_back((double)one_hot_[i][j] / word_sum);
				
				TF<<(double)one_hot_[i][j] / word_sum<<" ";
			}
			
		}
		TF<<endl;
	}
	
	TF.close();
}

//����ʻ�����ڶ��ٸ��ļ��У���ʱû�м����IDF 
void word_in_file_func()
{
	for(int i = 0; i < c.size(); i++)
	{
		for(int j = 1 ; j <= file_num; j++)
		{
			for(int k = 0; k < file[j].size(); k++)
			{
				if(c[i] == file[j][k])
				{
					word_exit_in_file[i+1]++;
					break;
				}
			}
		}
	}
	return;
 } 
 
 //����TF_IDF���� 
 void TF_IDF_func()
 {
 	word_in_file_func();
 	
 	ofstream TF_IDF("TFIDF.txt"); 
 	
 	for(int i = 1; i <= file_num; i++)
 	{
 		for(int j = 0; j < c.size(); j++)
 		{
 			TF_IDF << TF_[i][j] * log((double) file_num / (1 + word_exit_in_file[i]))<<" ";
		 }
		 TF_IDF<<endl;
	 }
	 TF_IDF.close();
 }
 
 //���ϡ�������Ԫ˳��� 
 void smatrix_func()
 {
 	ofstream smatrix("smatrix.txt");
 	//ǰ������Ϊ�ļ��������ʻ���С��one_hot������1������ 
 	smatrix << file_num << endl;
 	smatrix << c.size() << endl;
 	smatrix << word_num << endl;
 	
 	for(int i = 1; i <= file_num; i++)
 	{
 		for(int j = 0; j < one_hot_[i].size(); j++)
 		{
 			if(one_hot_[i][j] != 0)
 			{
 				smatrix << i - 1 << " " << j << " "<< one_hot_[i][j] << endl;
			 }
		 }
	 }
 	
 }
 
 //����ϡ�������ӣ��õ���Ԫ�������ʽ 
 void AplusB_func()
 {
 	ofstream AplusB("AplusB.txt");
 	int not_zero = 0;
 	for(int i = 1; i <= file_num / 2; i++)
 	{
 		int j = file_num / 2 + i;
 		for(int k = 0; k < c.size(); k++)
 		{
 			//ֵΪ��������� 
 			if(one_hot_[i][k] != 0 || one_hot_[j][k] != 0)
			 {
			 	smat[i-1].push_back(one_hot_[i][k] + one_hot_[j][k]);
			 	not_zero++;
			 }
			 //����ֵΪ0 
			 else
			 {
			 	smat[i-1].push_back(0);
			 }
		}
	}
	
	//������ļ��� 
	 AplusB << file_num / 2 << endl;
	 AplusB << c.size() << endl;
	 AplusB << not_zero <<endl;
	 
	 for(int i = 0 ; i < file_num / 2; i++)
	 {
	 	for(int j = 0; j < c.size(); j++)
	 	{
	 		if(smat[i][j] != 0)
	 			AplusB << i << " " << j << " " << smat[i][j] << endl;
		} 
	 }
	 
 } 

int main()
{	
	cout<<"vocabulary is creating!"<<endl; 
	vocabulary();  
	cout<<"one_hot matrix is creating!"<<endl; 
	one_hot_func();
	cout<<"TF matrix is creating!"<<endl; 
	TF_func();
	cout<<"TF_IDF matrix is creating!"<<endl; 
	TF_IDF_func();
	cout<<"smatrix is creating!"<<endl; 
	smatrix_func();
	cout<<"AplusB matrix is creating!"<<endl; 
	AplusB_func();
	cout<<"Finish!"<<endl; 

	return 0;
}
