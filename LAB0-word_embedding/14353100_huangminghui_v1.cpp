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
********使用哈希表***********
*****************************

#define HashTableSize 10000

string s[HashTableSize],t[HashTableSize]; 
int find_sum = 0;//查找次数 

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
	//元素的哈希值 
	 key = ELFHash(str) % (HashTableSize - 10);

	//如果这个值没有被占用，就直接映射到这个值上 
	if (HashTable[key] == NULL) 
	{
		HashTable[key] = (Node *)malloc(sizeof(Node));
		HashTable[key]->str = str;
		HashTable[key]->next = NULL;
		return;
	}

	tempA = HashTable[key];
	//如果这个值被占用，那就用拉链法，也就是在原有值的指针指向新的值 
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

//查找函数 
bool HashSearch(const char * str, Node ** HashTable)
{
	//输入的词的哈希值，用来查找 
	int key = ELFHash(str) % (HashTableSize - 10);
	Node * temp;

	//表为空或者这个值没有被占用，就是说这个单词不在单词表里 
	if (HashTable == NULL || HashTable[key] == NULL) 
	{
		find_sum++;//有没有找到都找了一次 
		return false;
	}
		
	temp = HashTable[key];
	//如果找到就返回指针，不然就一直找到值没有被占用来证明单词不在表里 
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
//初始化一个哈希表 
Node ** HashInit(int size)
{
	Node ** HashTable = (Node **)calloc(size, sizeof(Node *));
	for(int i = 0 ; i < HashTableSize; i++)
	{
		HashTable[i] = NULL;
	}
	return HashTable;
}

//整理一个词汇表	
void vocabulary() 
{
	vector<string> s;
	string line;						
	int n = 0,flag;
	
	ifstream in("semeval");

	//假如没有正常打开文件，就报错 
	if (!in.is_open())		
	{
		cout<<"Error opening file";
		exit (1);
	}
	
	//file_num用来计算文件数目 
	file_num = 0;
	Node ** HashTable = NULL;
    HashTable = HashInit(HashTableSize);
			
	while(!in.eof())	
	{
		string str;
		//从文件中取出一行 
		getline(in,line);
		
		//假如数据中有空行，则不用理会 
		if(line == "")
		{
			continue;
		}
		
		file_num++;
		
		//从第二个tab之后就是词汇了 
		int tab = line.find('\t',10);	
		//把词汇从一整句中分离 
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

	//放进c这个vector 里面 
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
********未使用哈希表***********
*****************************/
//整理一个词汇表	
void vocabulary() 
{
	vector<string> s;
	string line;						
	int n = 0,flag;
	
	ifstream in("semeval");

	//假如没有正常打开文件，就报错 
	if (!in.is_open())		
	{
		cout<<"Error opening file";
		exit (1);
	}
	
	//file_num用来计算文件数目 
	file_num = 0;
			
	while(!in.eof())	
	{
		string str;
		//从文件中取出一行 
		getline(in,line);
		
		//假如数据中有空行，则不用理会 
		if(line == "")
		{
			continue;
		}
		
		file_num++;
		
		//从第二个tab之后就是词汇了 
		int tab = line.find('\t',10);	
		//把词汇从一整句中分离 
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
	
	//得到词汇表c 
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

//检测一个文件中是否有某个词汇 
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

//one_hot矩阵 
void one_hot_func()
{
	ofstream one_hot("onehot.txt"); 
	
	//假如某个词汇出现，则输出1，否则输出0 
	//且记录这个词汇在该文件中出现次数，方便TF矩阵计算 
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

//计算TF矩阵 
void TF_func()
{
	ofstream TF("TF.txt"); 
	//输出TF矩阵之后，同时放进TF_这个vector记录，方便下面计算TF_TDF矩阵 
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

//计算词汇出现在多少个文件中，暂时没有计算出IDF 
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
 
 //计算TF_IDF矩阵 
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
 
 //输出稀疏矩阵三元顺序表 
 void smatrix_func()
 {
 	ofstream smatrix("smatrix.txt");
 	//前三个数为文件数量、词汇表大小和one_hot矩阵中1的总数 
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
 
 //两个稀疏矩阵相加，得到三元组矩阵形式 
 void AplusB_func()
 {
 	ofstream AplusB("AplusB.txt");
 	int not_zero = 0;
 	for(int i = 1; i <= file_num / 2; i++)
 	{
 		int j = file_num / 2 + i;
 		for(int k = 0; k < c.size(); k++)
 		{
 			//值为非零则相加 
 			if(one_hot_[i][k] != 0 || one_hot_[j][k] != 0)
			 {
			 	smat[i-1].push_back(one_hot_[i][k] + one_hot_[j][k]);
			 	not_zero++;
			 }
			 //否则，值为0 
			 else
			 {
			 	smat[i-1].push_back(0);
			 }
		}
	}
	
	//输出到文件中 
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
