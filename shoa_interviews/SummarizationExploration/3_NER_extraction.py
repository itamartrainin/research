#%%
import stanza
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

#%%
segments = pd.read_excel('C:\Data\shoa_dataset\Segmented_w_topics_gold_docs\joined_segments_tagged\segments.xlsx')
segments = segments[:10]

#%%
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

#%%
def ent_ext(doc):
    res = nlp(doc)
    return [(ent.text, ent.type) for ent in res.ents]
segments['entities'] = segments['content'].progress_apply(ent_ext)

#%%
for i, row in segments.iterrows():
    print(f'\n------{row["doc_ix"]}------\n')
    print(row['content'])
    for ent in row['entities']:
        print(ent[0] + ' - ' + ent[1])

#%%
text = """
####################
INTERVIEWER 1: Wait, wait, wait. What?.
INTERVIEWER 2: Your name?
INTERVIEWER 1: Your name first.
SUBJECT: My name.
INTERVIEWER 1: Wait, wait, wait till he tells me. OK. OK.
SUBJECT: Uh, I'm Joseph Zimmerman I was born Amer-- in-- Amer-- in Vienna in 1918. I was born in the house in which I lived until several months after Hitler came to Vienna. And I lived in that house through all the years that I went to school, public school, and then gymnasium five years, and then two years to medical school, again, until Hitler came to Austria and made me stop.
INTERVIEWER 1: Describe that apartment and your family.
SUBJECT: Uh, the apart-- the apartment was in the sixth district, which is one of the, uh, let's say, middle class districts of Vienna. And it was not, uh, heavily Jewish, uh, populated, such as the second or the 20th district in Vienna.
And, uh, for instance the people that lived in the same house, which i really feel were probably half and half. And actually, this was more Jewish than the neighbor house, for instance. So it was rather middle class, average type of Viennese circumstances.
We had, uh, four rooms and bathroom, the kitchen. The family consisted of my father and mother and, in addition to me, two sisters who were 1 and 1/2 of the 3 and 1/2 years younger than I was.
Uh, up until the very end when we left Vienna, we were all together. So that, uh, uh, there was a rather uninterrupted common experience in my family, whatever we went through. My father, uh, was a tailor who had his own store in the same district just a block away from where we lived. And, uh, I went to school uninterruptedly until Hitler came. And I did not, uh, do anything else. Uh, I helped my father some in the semester to learn part of his craft because he wanted me to do it. And, uh, at that particular time, I didn't care too much about it.

But subsequently after Hitler came to Austria, Vienna, for instance, and I had to interrupt my medical studies, and my father was able to continue to work in his own store, albeit with a sign outside, the Jewish store. And of course, no gentile client or customer would ever come in afterwards.
There was a lot to do because the Jews were trying to emi-- emigrate and could not take any money out of Vienna, for instance. We were all buying as much materials as they could and had suits made-- a very interesting experience. He couldn't get any more workers to work for him. So he needed my help and my sister's help and so on.
I became sort of a tailor, actually, the 1 and 1/2 years that I lived in Vienna after Hitler came. INTERVIEWER 1: Let's go back, though. Your friends in school, the Jewish, non-Jewish the kind of interaction you had with them and with their-- anything was different, let's say, in the early '30s, what you noticed.
SUBJECT: Well, let me tell you even going back further. Uh, I don't want to editorialize, but-- but anti-Semitism in Vienna did not wait for Mr. Hitler, OK? It-- to get-- it's always been there. I think it's-- it's endemic, like, like Wiener Schnitzel or Schlagobers. I mean, it's just part of the-- of the Austrian picture.
And so whether the Social Democrats were in power or the Christian Democrats were in power, it made no difference. My experience in Vienna with Gentiles was that they were, most of them, anti-Semitic from the beginning.
And I remember fights in parks where we were playing soccer from day age 10. Even before, in before in-- in-- in grammar school that the-- the-- the Christian boys were anti-Semitic in many instances and that the Jewish boy was always at a disadvantage and was always, uh, you know, fighting and there, particularly those that couldn't defend themselves were always in fights. So that anti-Semitism in Vienna was always present.

As times got worse, and particularly after social democrats got out of power-- and I remember 1927, I was nine years old. There was an enormous uprising by the social democrats.
And there was street fighting in Vienna with machine guns. And-- and, uh, it happened to be in the middle of the summer. And we were out in the country. And my father used to come out on weekends. There was a general strike. No-- no railroads were running. And this was 40 kilometers away from Vienna.
And my father walked out that weekend to see us. So the unrest were ready there at that particular time. And that get-- gets worse and worse now. In 1934, I remember the Dollfuss affair when the Nazis were shot Dollfuss in Vienna.
Dollfuss was the con-- the chancellor of Austria. And he was the-- the head of the Christian Democratic Front. And this was an attempt of the Nazis to take over the power of Austria. It was 1934.

In 1934, I remember the Dollfuss affair when the Nazis were shot Dollfuss in Vienna.
Dollfuss was the con-- the chancellor of Austria. And he was the-- the head of the Christian Democratic Front. And this was an attempt of the Nazis to take over the power of Austria. It was 1934.
By that time, I would say 90% of the students in gymnasium today went to-- were Nazis-- 90%. The-- the class consisted of about 34 students. Five of us were Jewish. And I think four or five were Christian democrats. There wasn't one socialist or communist in this class. All the others were Nazis, declared Nazis.
I remember-- I remember excursion, school excursions through the Vienna woods, which been-- had four or five per year, which was obligatory to go to. And when we marched down the-- the road, they would sing, uh, the \"Horst Wessel Song\" at the time when they really were supposed to be the Nazi party.
INTERVIEWER 1: Can you sing it? Can you remember it?
SUBJECT: Of course. I don't want to sing the \"Horst Wessel Song.\" \"Die Stra??e frei den braunen Bataillonen.\"
INTERVIEWER 1: No, no, the-- the music--
SUBJECT: \"Die Stra??e frei den braunen Bataillonen. Die Stra??e frei dem Sturmabteilungsmann. Es schau'n aufs Hakenkreuz voll Hoffnung schon Millionen. Der Tag f??r Freiheit und f??r Brot bricht an.\" I think that's about the beginning.
INTERVIEWER 1: Why didn't you want to sing that? What was so terrible?
SUBJECT: Well, to me, it is, uh, is a horror.
INTERVIEWER 1: But why?
INTERVIEWER 2: But you left out the stanza, which has specifically deals with Jews.
SUBJECT: \"Zum letzten Mal wird Sturmalarm geblasen.\" No, I don't-- I don't remember the part of the Jews. Do you remember?
INTERVIEWER 2: Wenn Judenblut von...
SUBJECT: But-- but that's not the \"Horst Wessel Song.\" That's the-- \"Wenn das Judenblut von Messer spitzt dann gehts nochmal so gut.\" But that is not the \"Horst Wessel Song.\" The \"Horst Wessel Song\" has to do with the takeover of power.
\"Zum letzten Mal wird Sturmalarm geblasen. Zum Kampfe steh'n wir alle schon bereit. Schon flattern Hitlerfahnen ??ber allen Stra??en. Die Knechtschaft dauert nur noch kurze Zeit.\" But I don't remember anyth-- any reference, in this particular song, to Jews.
INTERVIEWER 1: Why did you not want to sing it, then?
SUBJECT: You know, I have heard this song from, I would say, age 14 through the next seven years until I left Vienna. Under such unpleasant circumstances, so many times in my life, I don't-- I don't want to hear it. I don't want to sing it. And I certainly don't--
INTERVIEWER 1: Well, what un-- unpleasant circumstances did you hear this song?
SUBJECT: Well, you can imagine what the atmosphere is in-- in a class. And you know, for-- for people that you have to go to school with every day of your life, uh-huh. You also go with them on, for instance, ski class, ski courses.
We had twice, in two years, we went skiing in the Alps for a week. And some, uh, it was not-- that was not obligatory. It was just supposed to be a pleasure week. And you have this over-- overwhelming majority of Nazis in your classroom and maybe two or three Jews along with you. And they pick on the Jews.
They-- they take away the stuff. They annoy them. They-- they ridicule them. Now--
INTERVIEWER 2: Those of what ages?
SUBJECT: We're talking age 16, 17--
INTERVIEWER 1: Why--
SUBJECT: --big boys.
INTERVIEWER 1: Why did you go?
SUBJECT: We like to ski. Why should I not go? Why shouldn't I go?
INTERVIEWER 1: But that's--
SUBJECT: I-- actually, it was about the only thing that I did in common with those boys from age 13, I would think. But they were my classmates. And I did not, at that particular time, ever hope to see Hitler in Vienna. In fact, I remember one-- one excursion.
And you know, it is a very odd experience. I think American Jews don't know what Jews were like in Europe. And many-- that the emphasis here on physical health and competition and sports, and so on, was not as emphasized in Vienna, for instance, certainly not at in my father's time.
And in my time, I was different-- be-- beginning to have a more Jewish sports interest. I belonged to the Maccabee, and then-- and so on. But there were many, particularly Jews in gymnasium, for instance, who had absolutely no sports sense whatsoever.
They-- they weren't fighters. They weren't-- they just went to school to become lawyers or doctors and so on. But their-- their interaction with-- with the other guys was sort of on a-- on a level of your brainy, but I'm strong.
Now you have the other physical attitude now of the-- the mass of people, the overwhelming majority are-- are Nazis. And there was this little Jewish boy there who can't fight. He can't defend himself. They ridicule him.
So I was-- I really was different. I was-- I was in the sports field. I was one of the good ballplayers in the class. And I constantly had fights. I remember one thing there. One-- one of the ski excursions, when they took one of the boys who could never defend himself-- ridiculed him, bothered him day and night.
And I remember that I started fighting with them. I said, we are not in Nazi Germany yet. I'll never forget that thing. And that's what I thought we would never be. I didn't want to give up my rights to-- to-- to being a Viennese the way I wanted to see it. A Viennese, sure, live my own life because they-- they were in the majority in the-- on this class trip.


And so I had-- I had fights to-- all through those years, all through those years. And-- and I am, personally, made a decision in 1933. When Hitler went and took over Germany, I started boycotting every German product, every German film. I never went to a German film after 1933.
And I was not alone. Many other Jewish boys, you know, in their Zionist organizations and the-- the Maccabee. I did not got to any German films, didn't buy any German products. So we had very little trouble, in a way, when this finally happened to at least identify us where we belonged.
 But of course, there were other people. There was a lot of anti-Semitism, as I said, in Austria all the time. But there wasn't the religious basis. It was on a basis of the Jew. But many converted. Some, uh, people forgot that you were Jewish, you're now a good Catholic. It opened doors for you.
Many, many people I knew were-- were on that basis and-- and-- and were accepted, at least formally at that time, as-- as non-Jews. But when it happened to them, Hitler's laws came in, everything chu-- changed all of a sudden. And they realized that being a Christian on the-- on the conversion basis of one generation did not protect you.
And they had a very difficult time in coming to identify themselves and taking this punishment, which they didn't really feel. However, uh, the-- the good Christians that they converted to had no trouble whatsoever as identifying them as Jews when it came to take away their-- their stores and the property and so on in spite of the fact that they now were Catholic.


INTERVIEWER 2: What do you remember of the Anschluss or the Hinmarsch? What-- did--
SUBJECT: I remember that very vividly because if you remember, under the Christian-- After-- after the 1934 event, there where-- when Dollfuss was killed. Actually, all the parties in Austria were outlawed other than then the, uh, Christian Democratic party, or Christian Socialist party, of Dollfus and-- and, uh, Schuschnigg.
And many of the people, who were belonged to other parties, like before, uh, communists, even socialists, Shutzbund people for instance and Nazis were sitting in-- in detention camps already. And, uh, when this thing came that-- Hitler, obviously gave Schuschnigg the ultimatum, and we knew what was happening, all of a sudden, he permitted-- he let some of his prisoners free, including some of the socialists.
And for a very short week, very few days, you could see the emblems of the Socialist Party, it was-- it was something with three arrows, parallel arrows that they wore on the lapel, like the-- the Nazis wore the swastika. And were street fights going on all over Vienna between groups of young people and old people. And everything was-- but Hitler was standing at the border.
INTERVIEWER 1: What were you doing?
SUBJECT: I was out there with these three-- with these three, uh, Pfeile as a socialist on the street. We could-- we couldn't-- we still hoping that something-- nothing was happening because at this particular time, Schuschnigg did not want Hitler in there. Schuschnigg didn't want it.
And Austria really had what they think is a-- is a mission of its own. Austria is not, really, I think my feeling, a national German. Austria is a national Austrian. The Ostmark has a certain function to fulfill. And I think that Schuschnigg and Dollfuss really meant that, just like, uh, the Habsburgs had a mission separate from-- from-- from the German Reich.
And I think that Nazism in-- in Austria is not big-- did not become big because of the-- they wanted the-- the big German dream. I think they were sort of financially not able to survive. And anti--Set-- Semitism suited them well. But they're not really the-- the big-- uh, the big thing was not to become, uh, the Grosse German Reich originally.
So that there were people in Austria who didn't want Hitler to come in. And in those days, there was-- there were all kinds of things going on in the streets and on-- on-- on the pl-- on the-- on the, uh, squares of Vienna. But Hitler was there. And he marched in.
INTERVIEWER 1: But-- but you've-- your parent, your-- your father and mother. What did-- your sisters. What did they think about your being on the street and-- and--
SUBJECT: But-- but what everybody was, you know, but-- but the thing on the street, we didn't fight with weapons. We were just out there trying to protest and to-- to show that we didn't want Hitler to come in. I didn't have a weapon. We never had weapons. But I-- I can assure you that I was in street fights whether I wanted them or not.
INTERVIEWER 2: Other Jewish boys, too?
SUBJECT: Other Jewish boys, too? Well-- well-- well, particularly because we belonged there, we had a-- a big group of, uh-- belonged to Maccabee, uh, Turnverein. We had-- you played soccer in Jewish teams only.
I did-- you know, I played soccer week after week. And we had nothing-- and never had anything but the Jewish teams. And we never played another Jewish team. So we always played against the non-Jewish teams. And it-- it was always, you know-- always the-- the-- the possibility they-- there would be a fight.
INTERVIEWER 1: But the protests and the fights didn't help?
SUBJECT: They-- they could amount to nothing. The Austrian army was impotent against the German army, which was standing on the border. And Hitler just marched in.
INTERVIEWER 1: But before the-- the armies were involved, the-- you're saying that there was protest against injustice. There was protest against anti-Semitism. But it didn't mean anything?
SUBJECT: There was no protest against anti-Semitism. I do not recall saying that. I didn't say that. The-- the--
INTERVIEWER 1: No, the Jewish-- there were Jewish protests against--
SUBJECT: --laws at the time. The laws-- No good. Well, you have to defend yourself somehow OK.
INTERVIEWER 1: And that didn't help.
SUBJECT: No, it didn't help because--


INTERVIEWER 2: What was Schusch-- uh, Dollfuss and, uh, Schuschnigg's--
SUBJECT: Dollfuss and Schuschnigg were Christians.
INTERVIEWER 2: Christians in response to anti-Semitic.
SUBJECT: Well, I think that basically, I mean, uh, the whole concept of a Christian Democratic Party was-- was anti-- certainly not pro-Semitic. There weren't any members that I know of Jewish, uh, religion belonging to the party.
INTERVIEWER 1: But yet you were for them.
SUBJECT: Uh, I was not for them. It wasn't asked. It was a dictatorship just like any other. The Christian-- the Christian Socialist Party in-- in Vienna after Dollfuss' death was the only legal partry in the-- in the country. They did what they want. They didn't have any official, uh, anti-Semitics Semitic program.
You know, the laws still protected the Jew just as any other citizen. But a great love for-- for Jews, you do not see. For instance, there were-- there were appointments, uh, at the medical school, which-- which they very clearly showed the-- the tendency filling these jobs with followers of-- of-- of their party.
The anatomy professor in Vienna was by the name of [INAUDIBLE] was one of the most famous anatomists of all times left the country, certainly, I think, after-- in the-- in the early '30s because, uh, political trouble. And was certainly replaced with a Christian Democrat very promptly.
INTERVIEWER 1: I guess what I'm trying to ask you, Joe, is when the anti-Semiti-- Semitic behavior became very overt and the Jews, of course, reacted to some extent to protect themselves, did no one-- early, before Hitler came-- did no one in the non-Jewish community stand with them, helped them?
SUBJECT: There was no concerted pro-Semitic attitude in Austria at any time that I remember. There were members of the Socialist party which were Jewish, OK? And some of them became leaders in-- in the political field in Vienna in the first Democratic Party.
But it was not with the emphasis of-- of, uh, protecting the Jew. It was simply applied to-- to the general concept of political freedom. And-- and-- and that lasted as long as the real Social Democratic Republic of Austria existed.
INTERVIEWER 1: OK, but there was a very large Jewish community in Vienna. What was it, how many hundreds of thousands?
SUBJECT: There were 10% Jews approximately in Vienna.
INTERVIEWER 1: So what was that?
SUBJECT: But there was nothing of any political power that-- that really was concentrated on-- on helping the Jew, having a Jewish idea, or a Jewish identity, a Jewish party-- Nothing, nothing.
In fact, I think the Jews that were were were, uh, there-- there was a percentage which was very definitely strongly assimilated and wanted to have very little identification with the Jews. But, uh, they, though, did not convert, you know-- there was a very free-going attitude for-- for several years. And when-- when Hitler came to power, it--


INTERVIEWER 2: What was it like? Did they remember the moments the An-- Anschluss-- what happened?
SUBJECT: The Anschluss--
INTERVIEWER 2: --you see?
SUBJECT: The Anschluss was something that I didn't want to see personally because you couldn't be seen with a Jew on the street on that day. I mean, you-- you have to-- he came in from-- from the-- from the west. And they went down in his open Mercedes with, uh, the thousands of-- of-- of Nazi soldiers and that they-- they came in with.
It was a triumph. I think you can compare it only to what I've seen of the pope's visit here last year. I have never seen anything like this before or since.
INTERVIEWER 1: Where were you standing? Where were you looking?
SUBJECT: The Weg Gasse that I was at, that I lived at, is a side street of Mariahilfer Strasse You could hear the noise that was about a block-- two blocks away from the Mariahilfer Strasse.
You-- you couldn't help seeing that thousands and thousands of people, which were rushing to stand on the side and welcome him. The whole Mariahilfer Strasse down during-- there was no television in those days. You saw it subsequently, of course, in newspapers-- the pictures, everything was full of those pictures.
And, uh, I could hear the noise from where I lived. And it was the most triumphal reception of any conqueror that you ever want to see it he was conqueror. He was a very welcome conqueror at that time.
Nobody was fighting him at this time. It finished. There was no fight. The Austrian army did not fight. Schuschnigg did not let them fight. And there-- you know, there was no way to win this. And, uh, it was the end of, uh, of Austria.
INTERVIEWER 1: So what did you see? And what did you feel?
SUBJECT: Well, I had felt, for some time, that we were head-- we were heading for trouble, but I didn't really. I must tell you, honestly, ever, ever expected that he would just be able to walk in.
But he did. And there was nothing to be done. There-- there was no way that-- that-- that you could in any way resist this. This is a situation that one cannot, you know, you can't dream of it. It's-- it's a nightmare that-- that you, uh, that you suddenly have happened to you. And you hope that, uh, somehow how do you come out alive? You have no control of anything.


But no matter how bad you think it is, it was worse. Whatever you thought could be bad-- after all, we lived under dictatorship by Schuschnigg and Dollfuss for several years, OK. Jews didn't get any good jobs anymore. The-- the-- and government and so on.
Times weren't good. We knew that people with political objections were in-- in concentration camps or detention camps. We knew. So there's another dictator coming in. But that you suddenly, uh, completely naked and-- and-- and no law in the world will ever protect you against what anybody wants to do to you, that you have to experience. And that is-- that is the difference.
Anti-Semites, there always were fights-- we had all the time. Nobody protected our rights, particularly, OK? But that the law said that you were suddenly without any protection, that had never happened to us. And that you have to experience yourself, that suddenly in the same house in which you live, the landlord, whom you don't even-- I knew my landlord. He lived under me, OK?
He, uh, he was not a practicing Jew. In fact, I didn't know that he was Jewish. But the Nazis knew that he was Jewish. So they took the house away from him so fast you don't-- you wouldn't believe it.
The man was thrown out of his house and into the-- into the same apartment right under me, which is a very sizable apartment. It was about three times the size of my mine. The Nazis moved the office. So now all of a sudden, every time I go home, I have to pass by the Nazi office and all the traffic with all the Nazis with the SS and those [INAUDIBLE] uniforms in the same building. And I had to go up and down every time I went home.
And that went on. Uh, I don't know why they didn't throw us earlier out of the building because eventually they did 
But in, uh, in November when this--
INTERVIEWER 2: November--
SUBJECT: --Nazi-- November 9, when this Nazi attache--
INTERVIEWER 2: What--
SUBJECT: --was killed in Paris?
INTERVIEWER 2: --year? The-- '38?
SUBJECT: Pardon?
INTERVIEWER 2: 1938?
SUBJECT: 1938. Hitler has already been here for some months. Uh, that night-- I could go ba-- back further if you want. But it-- it's just-- just happened to think about this right now. Uh, it's been described over-- on the-- the 10th of November, they burnt all the temples in Vienna.
They arrested all the Jews that they could get hold. They picked them off the street. They made them, uh-- they made some old Jews and caftans, uh, scrub the-- the streets senselessly on their hands and feet with-- and they spit on them and everything else. They abused them in any-- any-- in any dehumanizing form that they-- they could do, OK?
INTERVIEWER 1: You saw this?
SUBJECT: Well, I didn't see it. I stayed home. I got wind of this early in the morning. And, uh, I just stayed home. I felt some how or other that they know me. This is-- they're right under me. They know where to get me.
And on the street, you are totally unprotected. From day one, from day one in Vienna, after Hitler marched in, every non-Jew wore a swastika in his lapel, whether he was an-- an organized Nazi from before or not. They made no difference whatsoever.
He wore the swastika. And if you didn't wear it, you were a Jew. Uh, you just had no chance on the 10th of November in Vienna on the street. You-- you were marked. So I felt maybe it's better to stay home.
Well, we stayed home all day. And we were waiting for them to pick us up. We didn't know it was going to-- at least they're not going to abuse us on the street. If they will come arrest us, I couldn't get out of this anyway.
But at 10 o'clock at night, the bell rang. And we mens didn't go to the door. The mother went to the door. And we expected them to say, we-- we've come to pick up your men. So she opens the door. And she said-- she said... Heil Hitler. My mother-- mother is suddenly gr-- grieving.
Uh, and they just said, is there anybody here wants to participate in the action? Would you believe that? I was born in this house. I lived in this house. They had come earlier some other evening before-- four or five SA men and said, we're going to take your piano, which they did.
But see, they do-- so they knew that we were Jewish. And on that particular day, they didn't think of me to come and pick us up. And maybe my-- my father's name Johann Zimmerman which was a very nice beaut-- beautiful Austrian name as well as a nice Jewish name.
But it's unbelievable. They came to ask us whether we wanted to participate in the action. On the same day, they arrested I don't know how many of their friends.
INTERVIEWER 2: What did your mother answer?
SUBJECT: My mother said, no, they-- they're out. They're already out, she said.
INTERVIEWER 1: She understood that they thought you were not Jewish.
SUBJECT: Yeah, yeah.
INTERVIEWER 1: And she said you were--
SUBJECT: W-- they're already out.
INTERVIEWER 1: \"Out,\" meaning?
SUBJECT: Out participating or something, she said. But this-- but this, uh, you know, presence of-- of mind, she said. They're not here. They said, Heil Hitler, and went away. It's unbelievable because somebody in that office downstairs knew.
So these guys, they came around but just didn't know us. And-- and they went away again. By the next day, things b-- were cooling off. And I didn't go out for a couple of days longer. But the 10th of November blew over for us. So it's-- it's very, very unbelievable situation.
INTERVIEWER 1: Your friends were arrested, your--
SUBJECT: Friends were arrested. Family was arrested.
INTERVIEWER 1: How'd you know?
SUBJECT: Well, because they weren't here all of a sudden. They were gone. We never saw them again.
INTERVIEWER 1: They called you on the telephone or [INAUDIBLE]?
SUBJECT: No, I found out later on.
We went-- we went back to life again when this was over. I mean, we-- we couldn't participate in any public functions. But I went to the store. And people came to the store. I had personal friends come into the house.
INTERVIEWER 1: Did you go to school?
SUBJECT: School was finished as of day one. I had to leave university within a week or 10 days after Hitler came.
INTERVIEWER 2: How exactly did this come about? What was--
SUBJECT: It wasn't--
INTERVIEWER 2: --hear about the edict?
SUBJECT: The edict was that all Jewish students had to leave the university, period.
INTERVIEWER 2: Who dec-- who, uh, made it public? How did you hear it?
SUBJECT: I-- I-- I don't exactly remember. I think I received a letter that I had to come and pick up my-- my-- my records or something. And it was certainly also in the newspapers. But I don't remember exactly now how-- it was-- it was-- it was just finished as of day one. I could not go back to school.
They, uh-- some medical students who were in the last semester, they let them finish. They didn't let them go back to school. But they-- they sort of gave them quick exams and let them finish. Uh, but I was in the fourth semester. So it was just completely hopeless-- no way.
INTERVIEWER 2: I heard from, um, in some circumstance that Jewish students were allowed, however, only standing. Is that--
SUBJECT: Only what?
INTERVIEWER 2: Standing.
SUBJECT: No.
INTERVIEWER 2: In classes in very good circumstances.
SUBJECT: No, not at the university. Nowhere, nowhere in Vienna. The-- the schooling was finished for the Jews in Vienna. They subsequently permitted what they called Umschungskurse schools, where you were permitted to learn some new trades and so on.
And I took one of those courses, for instance, in-- in, uh, cutting shirts, men's haberdashery, which I also did. I did all sorts of things to keep myself busy. And I, as I said, I went to my father and then became a halfway decent tailor And I learned a lot of English. But, uh, I could not go back to school.

INTERVIEWER 1: Why did you learn a lot of English?
SUBJECT: Because I was going to go to America.
INTERVIEWER 1: How-- how did you know?
SUBJECT: Well, I was-- I was hoping to go to America. I had relatives here. And-- and, you know, as I said before, we had no-- I had no problem with realizing the seriousness of the situation.
When you-- when you suddenly realize that there's a decree that you have to turn in whatever you own of value to the government-- it's just the decree. You have to turn that-- all the gold you have, all the stamp collections that you have, all the good silverware that you have. And you have to stand in line to give it to them.
When it is possible for them to come in the middle of the night and say they'll take your grand piano and you take it out of your home, when you know you can no longer go to school, when you know that your father's store has a sign out there that says, \"Jewish store,\" when you know that as soon as there are no Jews left, your father won't have any-- any livelihood either-- and you can't become anything or get any job anywhere in the world. And you can't go back to school, you know you have to leave the country.
There's nothing to hope for, you know. This is not like, uh, life was in a shtetl there. This is completely out of-- of anything-- any law of protection. There's nothing for you here. So we knew. I knew that I had to leave.
INTERVIEWER 1: How did you think you were going to be able to get out?
SUBJECT: Well, they-- they-- they, uh, didn't stop you unless they had political reasons for arresting you or other legal-- other reasons for arresting you. They didn't stop you from leaving the country. But you had to have a way of getting out by having a way to get in, all right?
There were many people who left the country, so to speak, illegally. They did not have any entry permit into any other country. They-- they didn't have any visa to enter any other country. But they had a-- a passport to leave.
And so they left, some of them very early in the game, particularly people who had political problems. And some of the socialists that had been released from the-- from the concentration camps.
I think the first thing they did was leave the country even before Hitler came in. I think Hitler was hardly walking in when they walked out somewhere else, OK? So they, uh, that was one way to go.
And one of my cousins left very early. I-- I think the first week of it, he-- he went to Switzerland. And he got there I know. But my parents didn't want me to go. They wanted me to wait. They all-- we were all going to go to America. We have relatives in America.
So we waited for the-- for the affidavit first from America. Then we waited for the quota number for America. And by July of 1939, which is something like 14, 15, 16 months after Hitler came, we were able to get out.
In the meantime, of course, was the 10th of November. There were all the other possibilities-- possible run-ins, all the chances of being arrested. All-- It took 15 months, 16 months for us to get out-- finally did.
But in those 16 months, of course, I was preparing for living in America. That's why I learned English.


INTERVIEWER 1: But it's the way you got out, I remember.
SUBJECT: No, there was one way that I didn't-- when I didn't get out, which was even more exciting.
INTERVIEWER 1: Yeah. What happened?
SUBJECT: Yeah? Well, you know, before the 10th of November, my-- my mother was not going to let me go. We were going-- first of all, I was first listed on an illegal transport to Israel. This was very early in the beginning, very, very soon after Hitler came. Uh, the Haganah was organizing these-- these trips.
And many of my friends from Maccabee and also from Betar were on this trip, and so was I. And just as they were about to take off, the affidavit arrived from America. And now my father said, my god, you're going on the illegal transport. You don't know if we ever there, the Nazi will arrest you. Don't go. We'll all go to America.
And, of course, I gave in. And I did not go on this trip. After 40 years, I just saw my friends in Israel last year. I think I told you that. And-- and within three weeks after that, two of them were here. It was very, very-- I hadn't seen them for 40 years. They got there. And most of them are alive today.

Anyway, this transport didn't go on. Then came the 10th of November. And things really changed. Now, all of a sudden, they burnt the temple. So many other people had been arrested, all this experience of the 10th of November. The cruelty, and so on, was behind us.
And now even my mother had feelings that maybe I shouldn't, as a young man, stay. Well, there was an unusual occasion that, uh, the, uh, mother-in-law of a friend of ours was going to go to Switzerland and to her in-laws, to his-- to his son-in-law, who-- who was Bren-- was in Ba-- in, uh, in Basel.
And they had made all the arrangements for this elderly lady. She must have been 65, 70, uh, with a guide waiting for her on the border. And she was going to be meeting this man. It was at the border Freiburg And they would get out there as soon as she got there.
And all I had to do was travel with this nice old lady to this border station. And they would pick us up at 9 o'clock at night. The guide would take us across. This seemed to be so fantastic. I mean, here's a-- a son and the daughter, you know, taking a chance and sending the old mother.
And illeg-- everything must be absolutely foolproof for this. It's safe enough for children to go along. The joke was along with this 70-year-old lady. And we travelled. It was New Year's Day, I think, of 1939.

And we travelled the whole day, the whole-- the whole night and some-- and we get there to this station in the middle of the night. Get off-- no guide. The guide didn't show up.
She says, what we do now? I didn't really feel like going back to Vienna. So I said, we have one chance. Can you get to people in Basel? Just said, when we get to Basel on the German station, I can call them. So we drove a-- we-- we took another train from there.
It took us, I think, the-- the best part of the next day to get to Basel. And we got to the Basel German station somewhere in middle afternoon of the following day. And she called up. And her German-- and her son-in-law and daughter-in-law came from the Swiss side of Basel into the German [INAUDIBLE]. They had Swiss passports.
Still, we have no guide. So, uh, how do we get there? He can't get us-- he can't get his mother in. He can't get me-- unless we find a way to get into Switzerland alone-- no guide either.
It's a very interesting arrangement. There's a-- the-- the Swiss- the German have part of the-- of-- of the Bonhoeffer Railroad Station. The Swiss are on the other. There's something like a maybe 200 yards between the two stations. When the Germans let you out to check your passport, you can get, uh, on the Swiss also look at you. Some of them in the-- in-- in the, uh, train. And that's it.
Well, but you have to have a visa to get in Switzerland. So the control was over on the German side. I had a valid German passport with a J in it-- I was a Jew-- and so did she. And we decided-- it's-- it's a crazy idea, OK? That when the Germans let us get on, OK, if we got on fast, the Germans let us get on. Maybe once we're in Switzerland, the Swiss won't bother us because many Swiss have-- many people have gotten the Swiss and-- and been permitted to stay once they were in.


But this-- you have to get inside Switzerland. Well, we go up to the station and-- and-- and quickly tried to get on this train so that the Germans won't bother us since you have no papers to get out. I get on the train. I have to have this elderly lady off the-- on-- on the train, you know? It's that train that's moving.
I help her up. She gets up. And as we come by the Swiss, I'm still hanging on the outside. And they pull me off. Now if I had been alone, maybe I could have made it. I don't know. Anyway, the old-- the lady was on the-- on the train. And I was not.
INTERVIEWER 2: The Germans pulled you off or?
SUBJECT: The Swiss pulled me off. And, uh--
INTERVIEWER 1: But you weren't in Switzerland yet, no?
SUBJECT: No, on-- on the border. But this-- that-- the-- the two stations located at a-- like-- like, practically the same station. It's 100 yards. The train moves. And-- and-- and-- and by the time she got on, I couldn't make it inside the train.
INTERVIEWER 2: They pulled you off and checked the passport?
SUBJECT: They pulled me off. And now-- this was not the Germans, now. I'm talking to the Swiss. And-- and-- and the Swiss say, well, what are you trying to do? I said, well, I'm trying to leave Germany because I can't live there.
And do you have a Swiss visa? I said, no. I can't let you in. And I pleaded with them for 10, 15 minutes. Told them I had an affidavit. And I'm only waiting for my quota number to go to the United States. I showed them a copy of my affidavit.
There was absolutely no way they would let me in. And, uh, and I said, you know, what are my options? I mean, I have to return to Germany? The Gestapo's going to arrest me. Uh, I'm sorry. But we have already admitted so many of, uh, of, uh, you know, refugees. We are not permitting anybody anymore without a visa.
And, uh, I had to go back. And, uh, actually, I'm sorry. This is now 40 years, and I had forgotten. It was the Germans that pulled me off the train. The Germans pulled me off the train. I have to correct this. You have to have to-- when you edit it, you have to correct-- I've forgotten.
This is-- this is-- the Germans pulled me off the train.

And the-- this-- and-- and they-- they started interrogating me. What was I doing? I said, I'm Jew-- I'm a Jew from Vienna. have no future here, as you know. I want to emigrate to America. But I can no longer wait. I have nothing to do. I'm-- I'm going crazy. I want to leave.
They-- they examined my papers. Uh, my passport was in order. How much money do you have? I said 10 marks. That was all I was allowed to take. If I had 15, they would have arrested me. I had 10 marks.
INTERVIEWER 1: Did they search you?
SUBJECT: They searched me, but I had no more money left. Uh, and they said, OK, you want to leave? Ask the Swiss. That's how it was. The woman was on the train. I asked the Swiss.
So I know why I had to do it this way because I could not buy a ticket to go to Switzerland. I could not officially buy the ticket to go. So we had to get on the train even without the OK of the Germans, OK?
I got on the train. But I didn't have time to get on. They pulled me off. But the German was fine. He said, you know, fine. As long as I didn't break the laws and I had a pass, but he didn't care. You go talk to the Swiss. The Swiss turned me down.
 INTERVIEWER 2: And how did they let her through? She didn't have a visa--
SUBJECT: I don't know what happened to the woman. She was on the-- she was on the train. Now you have to understand that she's an elderly lady. The whole trip from one station to the other is-- is nothing [INAUDIBLE]-- her son-in-law was waiting over there. Maybe he put out 20,000 francs for her. I don't know. When you had money, you could have done it.
INTERVIEWER 1: You never saw her again?
SUBJECT: I never saw her again.


INTERVIEWER 2: So you walked, or you--
SUBJECT: I walked over to the Swiss, this 100 yards over to the Swiss and-- and tried to negotiate my way in. I could not do it. I had to go back to the German station. And the Germans let me back in. I thought they were going to arrest me.
I had absolutely no idea with 10 marks in my pocket what to do. I'm hundreds of miles away from Vienna and 10-- 10 marks to get to-- 10 marks-- I couldn't get to Vienna. This a very interesting experience because at that point, I decided whatever I had to, I'm going to do with the Nazis.
I asked the man, what should I do now? I can't, with 10 marks, get back to Vienna. He said, go to Fr-- Freiburg, Freiburg im Breisgau. It's, I don't know, 60 kilometers from Basel. And they have a Jewish community there. And maybe they will help you get back.
Well, the trip to Freiburg, I think, costs something like nine marks and 70 Rappen or something. I could just make that ticket. And I arrived in Freiburg at 2 o'clock in the morning.
Now I don't know Freiburg from anything. I've never been there in my life. And here, I'm in Nazi Germany with a Jewish passport and then-- and half of a-- half a mark in my pocket. And what do I do now?
So I went to the police station, would you believe it? Because if I walked around in the-- Freiburg in the middle of the night, as a-- as a vagabond, they would have arrested me anyway. So I walk to the police station. And I told them the whole story, which they readily believed me.
I said, well, I can't do anything-- we can't do anything for you tonight because that Jewish man, the Jewish office doesn't open until 10 o'clock in the morning. They gave me the address. Uh, I said, where should I sleep? They said, why don't you go to this auberge there next-door, some-- some-- some lodging house.
I went there at 3 o'clock in the morning, rang the bell. I had 50-- 50 Rappen in my pocket. I couldn't have paid the lady. Uh, and in the morning, I went to this agency. Well, there were hundreds of Jews in Freiburg who had been there for the same reason.
They came. The guide didn't show up. They tried a guide. They were arrested at the border. The guide took the money but never came. They could not go back. They came from Poland. They came from eastern Germany. They came from Vienna.
They were afraid to go back. They had-- they were wanted by-- by the Gestapo. They had political affiliations. They had all kinds of-- of worries. They could not go back. The people sitting there for months. I assure you. I went to the telegraph office. And I-- and I got my money from Vienna within the same day. And I went back home to Vienna.
INTERVIEWER 1: How'd you--
SUBJECT: They were-- what?
INTERVIEWER 1: How did you get home? Oh with the money.
SUBJECT: Yeah, they-- they-- they wired me the money. I could not have stayed there. It was such misery, even then. People got no place to go. Been there for months. And somehow other, at that particular time, the Nazis knew about it. And they-- they-- they let them try over and over. It was really, uh, a misery, uh, beyond belief.
So that-- that trip did not materialize. I went back. It was, uh, by that time, I think, the fourth or fifth of January 1939.

And in July, we, uh, we were able to leave for America.
INTERVIEWER 1: How?
SUBJECT: Well, the quota number came through. My father was in the Russian quota, fortunately. And eventually, we came out.
INTERVIEWER 2: [INAUDIBLE]--
SUBJECT: Since I was not 21 years old yet, I came in his quota number. My sisters were even younger.
INTERVIEWER 2: Via? How did you get to [INAUDIBLE]?
SUBJECT: Well, via, uh, Paris, uh, London and with an English boat. And it was, if you remember, the war broke out September 1, I think. I left Vienna on the 17th of July.

And when we got to Paris, I suddenly realized that boy is just around the corner.
I did not-- you know, you don't realize it. I mean, that-- that the press is controlled. You don't know what the rest of the world thinks of you. You don't know what the feeling is outside the country. But in Paris, it was obvious in the middle of July that-- that the next time around, there will be war.
Next time Hitler tries anything, there will be war. And there was within-- four weeks after I came to this country, the war was broke out. So, uh,
INTERVIEWER 1: What-- what do you think about all of this now? I mean, you-- you were in the American Army?
SUBJECT: Yeah.
INTERVIEWER 1: Why did they take you?
SUBJECT: I don't-- I have no-- I don't really know why they shouldn't have taken me. I don't know why they didn't take me when I wanted to go. I wanted, in 1940, to go into the American Army. They wouldn't accept me as a volunteer.
In 1940, beginning of '42, they finally got around to drafting me. And, uh, you should have certainly that by that time, they would have certainly screened me. I came to Fort Dix. It was in March. I had, uh, I was at that time going to City College. I quit college.
And, uh, I reported. And they sent me back. I mean, this has really got nothing to do with the Holocaust. But there's a very interesting experience with the American government. I, uh, why are you sending me back? You have not been cleared. You were classified as an enemy alien.
I said, I'm an Ameri-- I'm, uh, an Austrian Jew. I already wanted to volunteer two years ago. You wouldn't take me. Now you're calling me. I should assume that you guys screened me. They said, I'm sorry. The screening is not adequate. Uh, we have to call you back.
I had to go back to the Bronx. Uh, I would say, how long will this take? Uh, two, three weeks. Take a vacation. This was March 1942. They did not call me again until October. Uh, you ask me how come-- uh, why should they have taken me? They should have taken me in 1940, OK?
So I had to go through all this. And then, when my division went to Europe, now I've already been in the army for 18 months. Again, they ask me, the G-2 called me twice to ask me if I had any objections to fighting against the Nazis. [CHUCKLING] Why shouldn't they have taken me?


INTERVIEWER 2: And you participated in the European theatre?
SUBJECT: Yes, I was in the European theatre.
INTERVIEWER 2: Where?
SUBJECT: I was-- I landed in Normandy in D plus 4. And I had the grosse Knackes to be with them all the way to Magdeburg in an infantry division all the way through the war.
INTERVIEWER 2: Into Germany?
SUBJECT: Into Germany to the Elbe.
INTERVIEWER 1: Do they use your language skills?
SUBJECT: Yes, they used it, uh, in-- in a very-- in a very strange way, really. They-- I-- I wan-- I had-- had-- had- disappointing experiences in the army, which is an incredible thing. Probably 11 million people, and-- and nobody knowing anything from anything. It's-- it's excusable.
The way they used me, finally, was my division commander would not let me get out of the division. I was a good soldier. I was a smart boys. So they didn't want to let me go. When-- and finally, I also applied, for instance, to, uh, to, uh, the intelligence school in-- in, uh, I forget now. It was a very famous-- uh, maybe you remember. Uh, I forget the name.
The OSS was trained there and so on. And-- and some of my friends were in it. Uh, I had all the qualifications. And they were going to take me. My division commander said, if they can use you, I need you more in my division that's going to Europe.
So from that point of view, he used me and wouldn't let me get out and get into-- into-- into a better job. And when we got to Europe-- to England, for instance, they sent me to Intelligence school in London for a week, where I got, uh, information about the German army's structure and so on, interrogations and so on.
So when we finally went to-- into the war, and then I was attached to-- to the interrogation of prisoners, uh, that were attached to my regiment. And that's-- that's the way they used my skills and language. But I was, all that while, actually in intelligence platoon of a-- of the regiment.

INTERVIEWER 2: Did you interrogate Nazis?
SUBJECT: Yes.
INTERVIEWER 2: And what was it like for you?
SUBJECT: What was it like? Well, let me tell-- I interrogated Nazis under fire. The rifle company level and-- and-- and-- and-- and sleep trenches, marching them back from front line companies to battalion headquarters in the middle of the night with one rifle and six Nazi prisoners in front of you. That-- that's how I started it out.
Then-- then there's, uh, as the war went on and they-- they realized they could use me at a higher level, they moved me into regimental headquarters. And then I-- and then interrogated, for instance, uh, in the Bulge, when they-- when they had, uh, this situation with, uh, a whole brigade of-- of-- of, uh, Germans trained in, uh, in, uh, behind the line of warfare, who were-- had started in America would-- were in the American uniforms and dropped behind our lines.
Those were the people I interrogated. And that-- that was a difficult situation. I remember one guy, the first guy I interrogated was a ballet dancer, who had been with the Ballet Russe in New York.
Uh, his English was, compared to mine at that time, superior. He spoke absolutely beautiful American English. You couldn't, you know, if he hadn't been caught in the circumstance in which he was caught, he could have passed as-- as-- as an American soldier anywhere in the world.
He was caught at a roadblock. And there wasn't a doubt about who he was, so I had, uh, that much leeway in-- in knowing what I was dealing with. But these people were trained to raise havoc behind our lines and did it. There was other people. He was-- he was shot in Liege a few weeks later.
INTERVIEWER 2: Shot in what?
SUBJECT: He was shot by-- by our, as a spy, by the-- by the American Army. Those are people that I interrogated. Uh, but it's a different, you know-- once they get that far back, you're dealing with a different level.
But they interrogate them at the front lines. Now that's a different story. Then you're-- you're standing on the-- on the shell fire. The machine guns are shooting around you. You want to have on-the-spot information. But buddy's around the corner. You threaten them, of course, in many ways to get information.
INTERVIEWER 2: Did you ever tell them you were a Jew?
SUBJECT: No. I didn't have any reason to. It wasn't-- it wasn't a question of my having personal satisfaction. I was fighting America's war against the Germans as far as they are concerned. I had no need to tell them that I was a Jew.
Uh, I also had fears about what happened to me if they ever caught me because, uh, certainly with my accent, with my English, it was-- was obvious that-- what I was. And that would have been very critical for me, particularly if it's in the counterattack, suddenly, they could have overrun our prisoners and then find out that I had interrogated them.
And that was a very tricky situation for me. But it never happened.


INTERVIEWER 1: What do you think now about your escape from your Europe, your family, who you left behind?
SUBJECT: Well, you know, left behind, uh, left behind, uh, two uncles, two aunts, and-- and-- and, uh, and a cousin and my wife's husband and my wife's father in concentration camps. But they were all killed. They were all killed.
INTERVIEWER 1: Why didn't they try to leave?
SUBJECT: Well, let me see. My wife's, uh, father is a unique individual. They were converted. And he was not a Jew by religion. He was-- he was a Jew by racial laws and was eventually arrested and-- and-- and, uh, and killed at Auschwitz.
Uh, my uncles and aunts, of course, were Jewish and were on the Polish quota and could not get out. And I think this is the catastrophe. This is, uh-- you can quote me there-- this is the guilt of America that these people did not get out.
They had an affidavit at the same time as I had, maybe even a month earlier. They were on the Polish quota. They never got out. And they were-- I don't know how many of them never got out.
And the American government did not bend one finger to-- to alleviate this problem to get those people out. That is a catastrophe. And I think about my escape. I'm very lucky I survived it. I survived all the consequences of it.
I even survived the fact that they kicked me out of medical school. It wasn't easy. But that's nothing. But that-- that-- that-- nobody helped these people get out. Nobody cared. That's-- that-- that is, you know, the lesson of the Holocaust-- this is part of it-- this is part of it, that we didn't help.


INTERVIEWER 1: What do you worry about now?
SUBJECT: I worry, in general principle, about the one fact that, uh, anti-Semitism didn't originate with Hitler. It's always there somewhere. And-- and-- and, uh, there's always fertile ground for it. And whenever, uh, social or financial conditions get critical in the country and you're looking for-- for a scapegoat, this is everybody's scapegoat.
And they have to be very-- very aware of the fact that-- that-- that these things can happen anywhere at any time and even today. It's a don-- it's not-- the danger is not over. And I think-- I think the things that, uh-- if you ask me about what-- what-- what happened in Vienna, didn't the Jews organize, OK? There was none. There was nothing.
There was no awareness of this. There was a willingness and a hope to-- to-- to disappear in-- in-- in-- in the-- in the crowd of, uh, of, uh, of Aryan Austrian to-- to-- to become as Austrian-- more Austrian than the next.
And I tell you. It's very tempting. It's a beautiful country. It's a beautiful culture. It's-- it's fantastic music. It's good food. It's-- it's-- it's all you want. But you can't become what they won't let you become.
And this-- and the awareness of what you are was brought home to all these people by-- by-- by, strangely enough, by the racial laws of Hitler. And you have to be aware that somebody can come in here tomorrow and decree the same way and that the rest of the county is not going to speak up for you. They're not. They're not.
And there were 10% Jews in Vienna. There was no concerted effort to, uh, to do anything. They-- they were-- some of them are trying to disappear. Some of them were having their own, uh, clubs and-- and-- and sports organizations and meetings and so on. But they actually started to really flourish after Hitler took over Germany.

INTERVIEWER 1: Did-- did Jews who were more identified with their Jewishness in Vienna-- did they fare better than those who didn't, in your experience?
SUBJECT: Well, I think I made that point before, that the ones who had the identification, at least, were a prior to leave because they felt it-- it was finished. My-- my-- my-- my uncle, for instance, felt that he was going to-- the one who couldn't leave anyway-- but he had money enough to get out. He had a lot of money.
He, uh, he could have bought his way out into some other country. And so many other people did who couldn't make it to America. They made it to Shanghai. They made it to South America. They made it all over the world with a lot of money. And he had a lot.
He sent some of his money off with his friends. But he was going to outlive Hitler, would you believe it? He didn't leave. And certainly, many people who were-- who were really not Jewish by religion didn't leave. They just didn't have any place to go. They had no identity with anybody else.
We had family in America, OK. So many of us had family in America. People whose parents came from Poland to Russia, at the time when they went to Vienna, some of the family when they went America.
That was the case in my fam-- in my family, that many of my father's brothers-- my father's sister and-- and two sisters of mother's had gone to America at that time. And my mother came to Vienna, for instance. That goes back to 1911.






"""
ents = ent_ext(text)

#%%
from collections import Counter
per_ents = []
for ent in ents:
    if ent[1] == 'PERSON':
        per_ents.append(ent[0])
        # print(ent[0] + ' - ' + ent[1])
print(Counter(per_ents))
