import argparse
from functools import partial
from pathlib import Path
import logging
import random
import sys
import regex
from datasets.utils.logging import set_verbosity_info
from datasets import load_dataset, load_from_disk

set_verbosity_info()
logger = logging.getLogger(__name__)
high_risk_tags = {'KEY', 'EMAIL', 'USER', 'IP_ADDRESS'} # , 'NUMBER', "ID"}
year_patterns = [
  # yyyy-yyyy or yyyy/yyyy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/][1-2][0-9]{3})(?:$|[\s@,?!;:\'\"(.\p{Han}])"), 
  # yyyy-mm-dd or yyyy-dd-mm or yyyy/mm/dd or yyyy/dd/mm or yyyy.mm.dd or yyyy.dd.mm
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/.][0-3][0-9][\p{Pd}/.][0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"), 
  # mm-dd-yyyy or dd-mm-yyyy or mm/dd/yyyy or dd/mm/yyyy or mm.dd.yyyy or dd.mm.yyyy or the same but with yy instead of yyyy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/.][0-3][0-9][\p{Pd}/.](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"), 
  # mm-yyyy or mm/yyyy or the same but with yy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"), 
  # yyyy-mm or yyyy/mm
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}-[0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"), 
]

# Patterns for high-risk character strings
id_pattern = r'(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([A-Za-z]*(?:[\p{Pd}]*\p{Nd}){6,})(?:$|[\b\s@?,!;:\'\")(.\p{Han}])'

# https://regex101.com/r/JQkmh8/5
key_pattern = r'(?:^|[\b\s@?,!:;\'\")(.\p{Han}])((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:_]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,}|(?:(?:\p{Nd}{3,}|[A-Z]+\p{Nd}+[A-Z]*|\p{Nd}+[A-Z]+\p{Nd}*)[ \p{Pd}]?){3,})(?:$|[\b\s\p{Han}@?,!;:\'\")(.])'

ipv4_pattern = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'
ipv6_pattern = r'(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
ip_pattern = r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + r"|".join([ipv4_pattern, ipv6_pattern]) + ")(?:$|[\s@,?!;:\'\"(.\p{Han}])"

# https://regex101.com/r/EpA5B7/1
email_pattern = r'''
    (?<= ^ | [\b\s@,?!;:)('".\p{Han}<] )
    (
      [^\b\s@?!;,:)('"<]+
      @
      [^\b\s@!?;,/]*
      [^\b\s@?!;,/:)('">.]
      \.
      \p{L} \w{1,}
    )
    (?= $ | [\b\s@,?!;:)('".\p{Han}>] )
'''

# https://regex101.com/r/mOqi1s/3
user_pattern = r'''
  (?<= ^ | [)(\s@,?!;:'"\p{Han}] )
  (@
    [^)(\s@,?!;:'"]{3,}
  )
'''
# Examples from https://regexpattern.com/phone-number/
# https://regex101.com/r/lZZ0XP/4
# Also matches MLS numbers
# phone_pattern = r'(?:^|[\s\'\"(\p{Han}])((?:\+\p{Nd}+[ \/.\p{Pd}]*)?(?:(?:\(\+?\p{Nd}+\))?(?:[ \/.\p{Pd}]*\p{Nd})){7,}(?:[\t\f #]*\p{Nd}+)?)(?:$|[\s@,?!;:\'\"(.\p{Han}])'

id_regex = regex.compile(id_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
key_regex = regex.compile(key_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
ipv4_regex = regex.compile(ipv4_pattern)
ipv6_regex = regex.compile(ipv6_pattern)
ip_regex = regex.compile(ip_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
email_regex = regex.compile(email_pattern, flags=regex.MULTILINE|regex.VERBOSE) #, re.MULTILINE)
user_regex = regex.compile(user_pattern, flags=regex.MULTILINE|regex.VERBOSE) #, re.MULTILINE)
# phone_regex = regex.compile(phone_pattern, flags=regex.MULTILINE) #, re.MULTILINE)



mst_regexes = {}
for tag in high_risk_tags:
  if tag == 'ID':
    mst_regexes['ID'] = id_regex
  elif tag == 'KEY':
    mst_regexes['KEY'] = key_regex
  elif tag == 'IPv4':
    mst_regexes['IPv4'] = ipv4_regex
  elif tag == 'IPv6':
    mst_regexes['IPv6'] = ipv6_regex
  elif tag == 'IP_ADDRESS':
    mst_regexes['IP_ADDRESS'] = ip_regex
  elif tag == 'EMAIL':
    mst_regexes['EMAIL'] = email_regex
  elif tag == 'USER':
    mst_regexes['USER'] = user_regex
#  elif tag == 'NUMBER':
#    mst_regexes['NUMBER'] = phone_regex
  else:
    sys.stderr.write('Dont have tag regex pattern for %s =(' % tag)

def ip_has_digit(matched_str):
  """Checks to make sure the PII span is not just :: or whatever that may
  accidentally be picked up by making sure there are digits."""
  return any(map(str.isdigit, matched_str))

def matches_date_pattern(matched_str):
  # Screen out date false positives
  for year_regex in year_patterns:
    if year_regex.match(matched_str):
      return True
  return False


def detect_pii(text, lang, tag_types):
  matches = []
  for tag in tag_types:
    label_pattern = mst_regexes[tag]
    # !! regex.match happens here!!
    matches_tmp = label_pattern.finditer(text)
    for match in matches_tmp:
      if match.groups():
        if len(match.groups()) > 1 and match.groups()[1]:
          sys.stderr.write("Warning: Found substring matches in the main match.")

        matched_str = match.groups()

        matched_str = matched_str[0]
        if matched_str:
          if tag in ["IP_ADDRESS"]:
            # Filter out false positive IPs
            if not ip_has_digit(matched_str):
              continue
          if tag in ["ID", "IP_ADDRESS"]: #, "NUMBER"]:
            # Filter out date false positives
            if matches_date_pattern(matched_str):
              continue
         
          matches += [(matched_str, match.span(), str(label_pattern), tag, lang)]
  return matches


#@title Redaction function defined here.
def redact_pii(text, matches):
  """Takes a match as defined in the detect_pii function and redacts it from the full string, returning a <redacted text, metadata> tuple."""
  redacted_str = text
  metadata = []
  for match in matches:
    matched_str = match[0]
    tag = match[3]
    redact_tag = "PI:" + tag
    redacted_str = redacted_str.replace(matched_str, redact_tag)
    # Create the "metadata" as all of the information we had before redaction
    metadata += [(match)]
  return (redacted_str, metadata)

#@title General function to run the PII detection and redact it, saving everything else to metadata, is defined here.
def run_pii(text, lang):
  """
  Runs the given set of regexes on the data "lines" and pulls out the
  tagged items.
  The lines structure stores the language type(s). This can be used for
  language-specific regexes, although we're dropping that for now and using
  only "default"/non-language-specific regexes.
  """

  text = text.encode().decode()
  matches = detect_pii(text, lang, high_risk_tags)
  match_set = (text, {})
  if len(matches) > 0:
    # !!! REDACTION HAPPENS HERE !!!
    redacted_str, metadata = redact_pii(text, matches)
    metadata_out = {"regex metadata":metadata, "original": text, "redacted": redacted_str}
    match_set = (redacted_str, metadata_out)
  return match_set


def run_pii_batch(exs, lang, text_column):
    """
    Runs the given set of regexes on the data "lines" and pulls out the
    tagged items.
    The lines structure stores the language type(s). This can be used for
    language-specific regexes, although we're dropping that for now and using
    only "default"/non-language-specific regexes.
    """
    regex_metadata = []
    old_text = []
    new_text = []
    modified = []
    for text in exs[text_column]:
        text = text.encode().decode()
        matches = detect_pii(text, lang, high_risk_tags)
        if len(matches) > 0:
            # !!! REDACTION HAPPENS HERE !!!
            redacted_str, metadata = redact_pii(text, matches)
            regex_metadata.append(repr(metadata))
            old_text.append(text)
            new_text.append(redacted_str)
            modified.append(True)
        else:
            regex_metadata.append("")
            old_text.append(text)
            new_text.append(text)
            modified.append(False)
    result = {
        "regex_metadata": regex_metadata,
        "old_text": old_text,
        "modified": modified
    }
    
    result[text_column] = new_text
    return result

def get_args():
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--text_column', type=str)
    parser.add_argument('--load_from_disk', action="store_true")
    parser.add_argument('--save_to_json', action="store_true", default=True)
    parser.add_argument('--dataset_path', type=Path)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument("--num_proc", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--save_batch_size", type=int, default=10000)
    args = parser.parse_args()
    return args

def get_check_ds(ds, args):
    if args.check_only_modified:
        ds_checks = ds.filter(
            lambda exs: exs["modified"],
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc
        )
    else:
        ds_checks = ds
    idx_samples = random.sample(range(len(ds_checks)), min(len(ds_checks), args.check_sampling_size))
    ds_checks = ds_checks.select(idx_samples)

    return ds_checks


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")
    file_path = Path(args.input_path)
    args.dataset_path=file_path.parent
    args.dataset_name=file_path.name
    logger.info(f" ===== Loading {args.dataset_path} =====")
    if args.load_from_disk:
        ds = load_from_disk(str(args.dataset_path))
    else:
        ds = load_dataset(str(args.dataset_path), data_files=[f"*{args.dataset_name}"], split="train")
    lang = str(args.dataset_path).split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]
    logger.info(f"ds info: {ds}")
    logger.info(f" ===== Applying PII =====")
    ds = ds.map(
        partial(run_pii_batch, lang=lang, text_column=args.text_column),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )

    ds_final = ds.remove_columns([
        "regex_metadata",
        "old_text",
        "modified"
    ])
    logger.info(f"ds_final info: {ds_final}")
    
    logger.info(f" ===== Saving Final dataset =====")
    logger.info(f"Saving to final dataset at {args.output_path}.")
    tmp_save_path = Path(args.output_path.parent, f"tmp-{args.output_path.name}")
    if len(ds_final) == 0:
        logger.info("Dataset was empty. Not saving anything.")
    else:
        if args.save_to_json:
            ds_final.to_json(
                tmp_save_path,
                num_proc=args.num_proc,
                batch_size=args.save_batch_size,
                force_ascii=False
            )
        else:
            ds_final.save_to_disk(tmp_save_path)
        tmp_save_path.rename(args.output_path)
        logger.info(f" ===== Final dataset saved successfully =====")
    '''
    ds_checks = get_check_ds(ds, args)

    logger.info(f" ===== Saving check dataset =====")
    logger.info(f"Saving check dataset at {args.save_check_path}.")
    tmp_save_path = Path(args.save_check_path.parent, f"tmp-{args.save_check_path.name}")
    if len(ds_checks) == 0:
        logger.info("Dataset was empty. Not saving anything.")
    else:
        if args.save_check_to_json:
            ds_checks.to_json(
                tmp_save_path,
                num_proc=args.num_proc,
                batch_size=args.save_batch_size
            )
        else:
            ds_checks.save_to_disk(tmp_save_path)
        tmp_save_path.rename(args.save_check_path)
        logger.info(f" ===== Check dataset saved successfully =====")
    '''