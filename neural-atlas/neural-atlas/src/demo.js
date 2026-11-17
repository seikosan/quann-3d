export const demo={name:'Mini Transformer',description:'Bir attention bloğu, üç paralel izdüşüm ve iki atlama bağlantısı içeren örnek mimari.',nodes:[
{id:'input',label:'Girdi',type:'input',shape:[8,1],description:'Sekiz öğelik örnek girdi dizisi.'},
{id:'embed',label:'Embedding',type:'embedding',shape:[8,48],params:1536,description:'Her girdiyi sayılardan oluşan bir temsile dönüştürür.'},
{id:'norm1',label:'Layer Norm',type:'norm',shape:[8,48],params:96,description:'Katmana giren sayıların ölçeğini düzenler.'},
...['Q','K','V'].map(x=>({id:x,label:x+' projection',type:'linear',shape:[8,48],params:2352,description:'Dikkat hesabında kullanılan '+({Q:'sorgu',K:'anahtar',V:'değer'}[x])+' temsili.'})),
{id:'attn',label:'Self Attention',type:'attention',shape:[3,8,8],description:'Dizideki öğeler arasındaki ilişkileri hesaplar. Üç dikkat başlığı gösterilir.'},
{id:'add1',label:'Residual Add',type:'add',shape:[8,48],description:'Dikkat çıktısını önceki temsille toplar. Atlayan bağlantı önceki bilgiyi taşır.'},
{id:'norm2',label:'Layer Norm',type:'norm',shape:[8,48],params:96,description:'İleri beslemeli bloğun girişini düzenler.'},
{id:'up',label:'MLP · genişlet',type:'linear',shape:[8,192],params:9408,description:'Temsili daha geniş bir ara uzaya taşır. Bu örnekte aktivasyon işlemi ayrıca çizilmemiştir.'},
{id:'down',label:'MLP · daralt',type:'linear',shape:[8,48],params:9264,description:'Genişletilen temsili başlangıç boyutuna geri getirir.'},
{id:'add2',label:'Residual Add',type:'add',shape:[8,48],description:'MLP sonucunu bloğun girişindeki bilgiyle birleştirir.'},
{id:'output',label:'Çıktı',type:'output',shape:[8,32],params:1568,description:'Örnek sözlük için her konumda 32 çıktı skoru.'}
],edges:[['input','embed'],['embed','norm1'],...['Q','K','V'].map(x=>['norm1',x]),...['Q','K','V'].map(x=>[x,'attn']),['attn','add1'],['embed','add1'],['add1','norm2'],['norm2','up'],['up','down'],['down','add2'],['add1','add2'],['add2','output']].map(([source,target])=>({source,target}))};
export const mlp={name:'Basit ileri beslemeli ağ',description:'Kendi katman tanımlarınla başlayabileceğin küçük bir örnek.',nodes:[{id:'x',label:'Girdi',type:'input',shape:[16]},{id:'h',label:'Gizli katman',type:'linear',shape:[32],params:544},{id:'a',label:'ReLU',type:'custom',shape:[32],description:'Negatif değerleri sıfırlar.'},{id:'y',label:'Çıktı',type:'output',shape:[4],params:132}],edges:[{source:'x',target:'h'},{source:'h',target:'a'},{source:'a',target:'y'}]};
