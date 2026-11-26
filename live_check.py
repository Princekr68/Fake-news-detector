import os, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["TOKENIZERS_PARALLELISM"]="false"
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)
import argparse
import concurrent.futures
import datetime as dt
import html
from html.parser import HTMLParser
import ipaddress
import json
import os
import re
import socket
import unicodedata
from urllib.parse import urlencode, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import xml.etree.ElementTree as ET

SEARX = os.getenv("SEARX_URL", "http://127.0.0.1:8888/search")
FEEDS = ("https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
         "https://www.thehindu.com/news/national/feeder/default.rss",
         "https://indianexpress.com/section/india/feed/",
         "https://feeds.feedburner.com/ndtvnews-india-news",
         "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
         "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3")
STOP = set("a an the to for of from is are was were will would have has had by at on in and it that did do does".split())
ALIAS = {"beats":"defeat","beat":"defeat","defeated":"defeat","defeats":"defeat",
         "lands":"land","landed":"land","wins":"win","won":"win","finals":"final","twenty20":"t20"}
UNCERTAIN = re.compile(r"\?|\b(?:if|might|could|whether|aims?|goal|purpose|predictions?|preview|hopes?|rumou?r|allegedly)\b",re.I)
LABEL = {"entailment":"REAL / SUPPORTED","contradiction":"FAKE / CONTRADICTED","neutral":"UNVERIFIED"}


def tokens(text):
    text = unicodedata.normalize("NFKC",text).lower().replace("’","'").replace("men's","mens").replace("women's","womens")
    text = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text)
    return [ALIAS.get(t,t) for t in text.split() if t not in STOP]


def overlap(claim, passage):
    a, b = set(tokens(claim)), set(tokens(passage))
    return len(a & b) / max(1,len(a))


def host(url):
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def public(url):
    u = urlsplit(url)
    if u.scheme not in ("http","https") or not u.hostname or u.username or u.password or u.port not in (None,80,443):
        raise ValueError("Invalid public article URL")
    addresses = socket.getaddrinfo(u.hostname,u.port or (443 if u.scheme=="https" else 80))
    if any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Expected a public article URL")


class Redirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        public(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def download(url,local=False):
    if not local:
        public(url)
    with build_opener(Redirect()).open(Request(url,headers={"User-Agent":"Mozilla/5.0 (college claim verifier)"}),timeout=9) as r:
        data=r.read(3_000_001)
        if len(data)>3_000_000:
            raise ValueError("Page exceeds size limit")
        return data.decode(r.headers.get_content_charset() or "utf-8",errors="replace"), r.geturl()


class Article(HTMLParser):
    """Keep headings and paragraphs; prefer blocks explicitly inside article/main."""
    def __init__(self):
        super().__init__()
        self.stack=[]
        self.blocks=[]
        self.active=None
        self.date=""

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="meta" and (attrs.get("property") or attrs.get("name")) in ("article:published_time","datePublished","date"):
            self.date=attrs.get("content","")[:10]
        if tag in ("article","main","nav","footer","aside","script","style","header","h1","p"):
            if tag in ("h1","p"):
                self.flush()
                self.active=(tag,any(t in self.stack for t in ("main","article")),[])
            self.stack.append(tag)

    def handle_data(self,data):
        if self.active and not set(self.stack)&{"script","style","nav","aside","footer"}:
            self.active[2].append(data)

    def flush(self):
        if self.active:
            tag,main,parts=self.active
            value=" ".join("".join(parts).split())
            if len(value)>=12:
                self.blocks.append((tag,main,value))
            self.active=None

    def handle_endtag(self,tag):
        if tag in self.stack:
            if tag in ("h1","p","article","main"):
                self.flush()
            del self.stack[len(self.stack)-1-self.stack[::-1].index(tag):]


def article(url):
    markup,link=download(url)
    doc=Article()
    doc.feed(markup)
    doc.flush()
    use_main=any(main and tag=="p" for tag,main,_ in doc.blocks)
    paragraphs=[v for tag,m,v in doc.blocks if tag=="p" and (m or not use_main)]
    if not paragraphs:
        raise ValueError("No readable article paragraphs")
    title=next((v for tag,_,v in doc.blocks if tag=="h1"),"")
    return dict(title=title,paragraphs=paragraphs,link=link,source=host(link),published=doc.date)


def search(query):
    addr=SEARX+"?"+urlencode({"q":query,"format":"json"})
    raw,_=download(addr,local=True)
    return [(r.get("title",""),r["url"]) for r in json.loads(raw).get("results",[]) if r.get("url")]


def rss(url):
    raw,_=download(url)
    return [(i.findtext("title",""),i.findtext("link","")) for i in ET.fromstring(raw).iter("item")]


def parallel(fn,values):
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(fn,value) for value in values]
        for job in jobs:
            try:
                yield job.result()
            except (OSError,ValueError,ET.ParseError,TimeoutError):
                yield None


def candidate_urls(claim):
    neutral=" ".join(t for t in tokens(claim) if t not in ("never","not","no"))
    results=list(parallel(search,list(dict.fromkeys([claim,neutral+" report",claim+" fact check"]))))
    pool=[r for batch in results if batch for r in batch]
    for batch in parallel(rss,FEEDS):
        pool.extend(r for r in batch or [] if overlap(claim,r[0])>=0.3)
    ranked=sorted(pool,key=lambda r:overlap(claim,html.unescape(r[0])),reverse=True)
    urls=[]
    counts={}
    for _,url in ranked:
        domain=host(url)
        if not domain or url in urls or counts.get(domain,0)>=2:
            continue
        urls.append(url)
        counts[domain]=counts.get(domain,0)+1
        if len(urls)==12:
            break
    return urls, all(x is None for x in results)


class Verifier:
    """Joint premise/hypothesis classification; use the model's declared label order."""
    def __init__(self):
        self.model=None

    def predict(self,claim,passages):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self.model is None:
            name=os.getenv("LIVE_NLI_MODEL","MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
            self.tok=AutoTokenizer.from_pretrained(name)
            self.model=AutoModelForSequenceClassification.from_pretrained(name).eval()
        predictions=[]
        for start in range(0,len(passages),4):
            batch=passages[start:start+4]
            inputs=self.tok(batch,[claim]*len(batch),return_tensors="pt",padding=True,
                            truncation="only_first",max_length=512)
            with torch.inference_mode():
                probs=self.model(**inputs).logits.softmax(-1)
            for dist in probs:
                idx=int(dist.argmax())
                label=self.model.config.id2label[idx].lower()
                if label not in LABEL:
                    raise ValueError("Model needs named entailment, neutral and contradiction labels")
                predictions.append((label if float(dist[idx])>=0.55 else "neutral",float(dist[idx])))
        return predictions


# For claims about a completed event, a schedule or prediction is neutral.
# Keep past and future sentences separate even within one live-blog paragraph.
PROSPECTIVE = re.compile(
    r"\b(?:will|shall|plans? to|set to|expected to|scheduled to|yet to|"
    r"preparing to|due to|going to|ahead of|before the upcoming)\b", re.I
)
COMPLETED = re.compile(r"\b(?:\w+ed|beat|won|lost|took|gave|held|began|bought|sold|built|sent|was|were|has|have|had)\b", re.I)


def eligible_sentences(claim, paragraph):
    past = COMPLETED.search(claim) and not PROSPECTIVE.search(claim)
    for sentence in re.split(r"(?<=[.!?।])\s+", paragraph.strip()):
        if UNCERTAIN.search(sentence) or (past and PROSPECTIVE.search(sentence)):
            yield None  # Break adjacency across an excluded sentence.
        else:
            yield sentence


def excerpts(doc,claim):
    windows=[]
    prev=None
    for paragraph in doc["paragraphs"][:65]:
        for sentence in eligible_sentences(claim,paragraph):
            if sentence:
                windows.append(sentence)
                if prev and len(prev)+len(sentence)<850:
                    windows.append(prev+" "+sentence)
            prev=sentence
    # The quote is always article text. The retrieved title is context only.
    windows=[w for w in dict.fromkeys(windows) if 15<=len(w)<=1200 and not UNCERTAIN.search(w)]
    return sorted(windows,key=lambda w:(overlap(claim,w),-len(w)),reverse=True)[:3]


def role_order(claim,quote):
    # Resolve an explicit subject/verb/object relation only when the event
    # qualifiers are present and both named parties occur next to the verb.
    m=re.fullmatch(r"(.+?)\s+(defeated|beat|acquired|sued|appointed)\s+(.+?)\s+(?:in|during|at)\s+(.+)",claim.strip(". "),re.I)
    if not m or re.search(r"\b(?:never|not|might|didn't|failed|could|would|will|denied|claims?)\b",quote+" "+claim,re.I):
        return None
    left,verb,right,event=m.groups()
    event_words=set(tokens(event))
    words=tokens(quote)
    if not event_words<=set(words):
        return None
    for a,b,label in [(left,right,"entailment"),(right,left,"contradiction")]:
        terms=tokens(a)+tokens(verb)+tokens(b)
        # Match participants in the same sentence, preserving grammatical order.
        for sentence in re.split(r"(?<=[.!?।])\s+", quote):
            if not event_words <= set(tokens(sentence)):
                continue
            line=" "+" ".join(tokens(sentence))+" "
            if " "+" ".join(terms)+" " in line:
                return label
            # "X ended ... when they beat Y" links the subordinate subject
            # to the leading named party. Reject an intervening named party.
            pattern=(r"^"+re.escape(a)+r"\s+(?P<lead>[A-Za-z0-9 ’'\-]{1,110}?)"
                     +r"\s+when (?:they|it|he|she) (?P<action>[a-z]+) "+re.escape(b)+r"\b")
            m=re.search(pattern,sentence)
            if m and m["lead"].startswith(("ended their ", "ended its ", "made history ", "secured their ")) and b.casefold() not in m["lead"].casefold() and tokens(m["action"])==tokens(verb):
                return label
    return None


def exact_date_relation(claim, quote):
    """Compare dates only in otherwise identical standalone statements."""
    pattern = r"\b(\d{1,2}) (january|february|march|april|may|june|july|august|september|october|november|december)\b"
    normalize = lambda s: " ".join(s.lower().strip().rstrip(".").split())
    a, b = normalize(claim), normalize(quote)
    x, y = re.findall(pattern,a), re.findall(pattern,b)
    if len(x)==len(y)==1 and re.sub(pattern,"DATE",a)==re.sub(pattern,"DATE",b):
        return "entailment" if x==y else "contradiction"
    return None


def mentions_requested_day(claim, passage):
    for iso in re.findall(r"\b\d{4}-\d{2}-\d{2}\b",claim):
        try:
            day=dt.date.fromisoformat(iso)
        except ValueError:
            return False
        text=passage.lower().replace(",","")
        variants=[iso]
        for month in (day.strftime("%B").lower(),day.strftime("%b").lower()):
            variants.extend((f"{day.day} {month} {day.year}",f"{month} {day.day} {day.year}"))
        if not any(v in text for v in variants):
            return False
    return True


def verify(claim,docs,verifier):
    rows=[]
    pending=[]
    model_error=None
    for doc in docs:
        for quote in excerpts(doc,claim):
            ctx=doc["title"]+"\n"+quote
            if overlap(claim,ctx)<0.15 or overlap(claim,quote)<0.10:
                continue
            # A date in the publication metadata alone does not establish the event date.
            years=set(re.findall(r"\b(?:19|20)\d{2}\b",claim))
            if not mentions_requested_day(claim,ctx) or years-set(re.findall(r"\b(?:19|20)\d{2}\b",ctx)):
                continue
            row=dict(item={k:doc[k] for k in ("title","link","source","published")},
                     passage=quote,sim=overlap(claim,quote),kind="model",rule=None)
            direct=role_order(claim,quote) or exact_date_relation(claim,quote)
            if direct:
                rows.append(dict(row,label=direct,kind="explicit"))
            else:
                pending.append(row)
    # A headline cannot supply the vote. Only input the body quotation to NLI.
    explicit_urls={row["item"]["link"] for row in rows}
    # For a page with a direct role/date match, do not reclassify the same
    # evidence in larger overlapping windows. Other pages remain independent.
    pending=[r for r in pending if r["item"]["link"] not in explicit_urls]
    if pending:
        try:
            pairs=verifier.predict(claim,[r["passage"] for r in pending])
            for row,(label,score) in zip(pending,pairs):
                if label!="neutral":
                    rows.append(dict(row,label=label,score=score))
        except (OSError,ValueError,ImportError,RuntimeError) as err:
            model_error=f"NLI unavailable ({type(err).__name__}): {str(err)[:180]}"
    return rows,model_error


def decide(rows):
    # One source cannot outvote itself. Copies from multiple sites count once.
    rows=sorted(rows,key=lambda r:(r["kind"]=="explicit",r["sim"]),reverse=True)
    labels={r["label"] for r in rows}
    if len(labels)>1:
        # Majority voting: deduplicate by source first, then count ---
        seen_sources={}
        for r in rows:
            src=r["item"]["source"]
            # Keep the highest-overlap entry per source
            if src not in seen_sources or r["sim"]>seen_sources[src]["sim"]:
                seen_sources[src]=r
        votes=list(seen_sources.values())
        ent_votes=[v for v in votes if v["label"]=="entailment"]
        con_votes=[v for v in votes if v["label"]=="contradiction"]
        if len(con_votes)>len(ent_votes):
            return "contradiction","Majority of distinct sources contradict this claim.",con_votes[:3]
        if len(ent_votes)>len(con_votes):
            return "entailment","Majority of distinct sources support this claim.",ent_votes[:3]
        # Truly tied: show both sides for manual review
        conflict=[next(r for r in rows if r["label"]==label) for label in ("entailment","contradiction")]
        return "neutral","Sources are evenly split. Review both passages.",conflict
    unique=[]
    for r in rows:
        words=set(tokens(r["passage"]))
        if any(r["item"]["source"]==x["item"]["source"] or
               len(words & other)/max(1,len(words | other))>=0.78
               for x in unique for other in [set(tokens(x["passage"]))]):
            continue
        unique.append(r)
    if not unique:
        return "neutral","No readable passage directly settled this claim.",[]
    if len(unique)==1 and unique[0]["kind"]=="explicit":
        return unique[0]["label"],"One original passage explicitly states the relevant roles or date; see the quoted source.",unique
    if len(unique)<2:
        if unique and unique[0]["kind"]=="explicit":
            return unique[0]["label"],"One original passage explicitly matches.",unique
        if unique and unique[0].get("score",0)>=0.75:
            return unique[0]["label"],"One high-confidence report found.",unique
        return "neutral","One relevant report found; independent corroboration is missing.",unique
    return unique[0]["label"],"Matching article passages from distinct sources.",unique[:3]


def extract_from_url(url):
    """Extract the main claim from a URL (news article, social media post, etc.)."""
    try:
        doc=article(url)
    except (OSError,ValueError,TimeoutError):
        # For social media or JS-heavy pages, try a simpler approach
        try:
            markup,final_url=download(url)
            # Try to get Open Graph / meta tags
            title=""
            description=""
            for pattern,target in [
                (r'<meta\s+(?:property|name)=["\']og:title["\']\s+content=["\']([^"\']+)["\']',  "title"),
                (r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\']og:title["\']',  "title"),
                (r'<meta\s+(?:property|name)=["\']og:description["\']\s+content=["\']([^"\']+)["\']', "desc"),
                (r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\']og:description["\']', "desc"),
                (r'<meta\s+(?:property|name)=["\']description["\']\s+content=["\']([^"\']+)["\']', "desc"),
                (r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\']description["\']', "desc"),
                (r'<title[^>]*>([^<]+)</title>', "title"),
            ]:
                m=re.search(pattern,markup,re.I)
                if m:
                    if target=="title" and not title:
                        title=html.unescape(m.group(1)).strip()
                    elif target=="desc" and not description:
                        description=html.unescape(m.group(1)).strip()
            # Try tweet text pattern (for X/Twitter)
            tweet=re.search(r'<meta\s+(?:property|name)=["\']og:description["\']\s+content=["\']([^"\']+)["\']',markup,re.I)
            text=description or title
            if not text:
                return None
            return dict(
                title=title or "(no title)",
                claim=title+" . "+description if title and description and title.lower()!=description.lower()[:len(title)].lower() else text,
                source=host(final_url),
                url=final_url
            )
        except (OSError,ValueError,TimeoutError):
            return None

    # Build a claim from the article title + first key paragraphs
    title=doc["title"]
    # Pick the most informative paragraphs (first 2-3 that are factual)
    key_paragraphs=[]
    for p in doc["paragraphs"][:8]:
        if len(p)>=30 and not UNCERTAIN.search(p):
            key_paragraphs.append(p)
            if len(key_paragraphs)>=2:
                break

    if title and key_paragraphs:
        claim=title+" . "+key_paragraphs[0]
    elif title:
        claim=title
    elif key_paragraphs:
        claim=key_paragraphs[0]
    else:
        return None

    # Truncate to reasonable length for NLI
    if len(claim)>500:
        claim=claim[:500].rsplit(" ",1)[0]

    return dict(
        title=doc.get("title",""),
        claim=claim,
        source=doc.get("source",host(url)),
        url=doc.get("link",url),
        published=doc.get("published","")
    )


def live_check(claim,cv=None):
    original=claim.strip()

    # --- URL support: extract claim from a URL ---
    source_domain=None
    if re.match(r"https?://",original,re.I):
        extracted=extract_from_url(original)
        if extracted is None:
            return dict(verdict="UNVERIFIED",reason="Could not read the page at the given URL.",
                        rows=[],n_items=0,web_error=None,rss_failed=[],
                        model_warning=None,articles_read=0,
                        input_url=original,extracted_claim=None)
        claim=extracted["claim"]
        source_domain=extracted["source"]
        print(f"EXTRACTED CLAIM: {claim}")
        print(f"FROM: {extracted['source']} | Title: {extracted['title']}\n")
    else:
        claim=original

    today=dt.date.today()
    offsets={"today":0,"tomorrow":1,"yesterday":-1}
    claim=re.sub(r"\b(today|tomorrow|yesterday)\b",lambda m:(today+dt.timedelta(days=offsets[m[0].lower()])).isoformat(),claim,flags=re.I)
    urls,failed=candidate_urls(claim)
    # Exclude the original source domain from verification (no self-verification)
    if source_domain:
        urls=[u for u in urls if host(u)!=source_domain]
    docs=[d for d in parallel(article,urls) if d]
    # Also exclude docs from the same domain
    if source_domain:
        docs=[d for d in docs if d.get("source")!=source_domain]
    votes,warning=verify(claim,docs,cv or Verifier())
    label,reason,rows=decide(votes)
    if not docs:
        reason="No original article could be read." if urls else "No candidate articles found."
    result=dict(verdict=LABEL[label],reason=reason,rows=rows,n_items=len(urls),
                web_error="SearXNG search failed." if failed else None,rss_failed=[],
                model_warning=warning,articles_read=len(docs))
    if source_domain:
        result["input_url"]=original
        result["extracted_claim"]=claim
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("claim",nargs="*")
    args=parser.parse_args()
    user_input=" ".join(args.claim) or input("Claim or URL: ")
    result=live_check(user_input)
    # Show extracted claim if URL was given
    if result.get("input_url"):
        print(f"INPUT URL: {result['input_url']}")
        print(f"EXTRACTED CLAIM: {result.get('extracted_claim','')}\n")
    print("VERDICT:",result["verdict"])
    print("Reason:",result["reason"])
    print("Original articles read:",result["articles_read"])
    for r in result["rows"]:
        i=r["item"]
        print(f"\n{r['label'].upper()} | {i['source']} | {i['published'] or 'Publication date unavailable'}")
        print(r["passage"])
        print(i["link"])
    for key in ("web_error","model_warning"):
        if result[key]:
            print(result[key])
