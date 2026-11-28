# ruff: noqa: E501  (a data file: one message or query per line reads best)
"""Synthetic mailbox for the retrieval eval: ~150 messages, ~30 correspondents, planted facts.

Hand-written so that the golden answers are known exactly. Deliberate difficulties:
- superseded facts (a price, a date or an option pool that changed later in the conversation);
- name collisions (two people called Alex, two called Jordan, at different companies);
- near-duplicate threads (two quotes, two bookings, two orders) that differ in one detail;
- identifiers (booking codes, ticket and invoice numbers) that embeddings blur and keywords catch;
- paraphrase: queries rarely reuse the wording of the message that answers them;
- hard distractors: the same person and topic words, but an outdated or wrong answer;
- noise: small talk, out-of-office notes and newsletters from the same people.

Each message: (id, thread, sender, to, date, subject, body). The owner is owner@example.com.
Each query: (id, sender, subject, body, relevant ids, category). A query plays the role of a
new incoming email from `sender`; retrieval must find the past messages that answer it.
"""

OWNER = "owner@example.com"

# fmt: off
MESSAGES: list[tuple[str, str, str, list[str], str, str, str]] = [
    # --- Alex at Brightweb (web agency): quote revised, launch date, hosting ---
    ('bw-1', 'bw-quote', 'alex@brightweb.example', [OWNER], '2026-06-02', 'Website redesign quote', 'Hi! For the full redesign (8 pages, CMS, contact forms) our estimate is $4,500, delivered in six weeks.'),
    ('bw-2', 'bw-quote', OWNER, ['alex@brightweb.example'], '2026-06-03', 'Re: Website redesign quote', 'Thanks Alex. Could you add a blog section and a newsletter signup?'),
    ('bw-3', 'bw-quote', 'alex@brightweb.example', [OWNER], '2026-06-05', 'Re: Website redesign quote', 'With the blog and newsletter signup the revised total is $5,200. Same six-week timeline.'),
    ('bw-4', 'bw-quote', OWNER, ['alex@brightweb.example'], '2026-06-06', 'Re: Website redesign quote', "Great, $5,200 works. Let's go ahead."),
    ('bw-5', 'bw-launch', 'alex@brightweb.example', [OWNER], '2026-07-01', 'Launch planning', 'We are aiming to launch the new site on Monday 14 September, after a final review on the 10th.'),
    ('bw-6', 'bw-hosting', 'alex@brightweb.example', [OWNER], '2026-07-08', 'Hosting', "We'll deploy on Netlify; the yearly hosting plan is $190 and we handle the DNS switch."),
    ('bw-7', 'bw-chat', 'alex@brightweb.example', [OWNER], '2026-07-20', 'Team offsite', 'Our team is off at a retreat next week, responses may be slower than usual.'),
    # --- Maria at Lumen Design (logo) ---
    ('ld-1', 'ld-logo', 'maria@lumen-design.example', [OWNER], '2026-05-11', 'Logo proposal', 'The logo package includes three rounds of revisions and final files in SVG and PNG.'),
    ('ld-2', 'ld-logo', 'maria@lumen-design.example', [OWNER], '2026-05-12', 'Re: Logo proposal', 'We ask for a 30% deposit up front; the balance is due on delivery.'),
    ('ld-3', 'ld-invoice', 'maria@lumen-design.example', [OWNER], '2026-05-20', 'Invoice LD-2291', 'Please find attached invoice LD-2291 for the deposit, $660, payable within 14 days.'),
    ('ld-4', 'ld-logo', OWNER, ['maria@lumen-design.example'], '2026-05-25', 'Re: Logo proposal', 'We prefer the second concept, the teal one with the rounded wordmark.'),
    ('ld-5', 'ld-chat', 'maria@lumen-design.example', [OWNER], '2026-06-15', 'Portfolio', 'Would you mind if we featured your logo in our portfolio once it launches?'),
    # --- Dr. Chen at River Clinic (dentist): appointment moved ---
    ('rc-1', 'rc-appt', 'dr.chen@riverclinic.example', [OWNER], '2026-03-02', 'Appointment confirmation', 'Your cleaning and check-up is booked for 2 April at 9:00.'),
    ('rc-2', 'rc-appt', 'dr.chen@riverclinic.example', [OWNER], '2026-03-25', 'Appointment change', 'Dr. Chen is away on 2 April, so we moved your visit to 9 April at 10:30. Same room.'),
    ('rc-3', 'rc-insurance', 'dr.chen@riverclinic.example', [OWNER], '2026-03-26', 'Insurance details', 'We have your dental insurance on file: policy DN-448812 with Safeguard, 80% coverage for cleanings.'),
    ('rc-4', 'rc-xray', 'dr.chen@riverclinic.example', [OWNER], '2026-04-10', 'X-ray results', 'Your X-rays look good, no cavities. Next check-up recommended in six months.'),
    # --- Tom at Oak Residences (landlord): rent increase, boiler, parking ---
    ('oak-1', 'oak-rent', 'tom@oakresidences.example', [OWNER], '2026-04-01', 'Rent from June', 'As discussed, rent will go from $1,250 to $1,350 per month starting 1 June.'),
    ('oak-2', 'oak-rent', OWNER, ['tom@oakresidences.example'], '2026-04-02', 'Re: Rent from June', 'Understood. Can the payment date stay on the 5th of each month?'),
    ('oak-3', 'oak-rent', 'tom@oakresidences.example', [OWNER], '2026-04-03', 'Re: Rent from June', 'Yes, the 5th of the month is fine.'),
    ('oak-4', 'oak-boiler', 'tom@oakresidences.example', [OWNER], '2026-10-14', 'Boiler repair', 'The heating engineer will come on Thursday between 8 and 11 to repair the boiler.'),
    ('oak-5', 'oak-parking', 'tom@oakresidences.example', [OWNER], '2026-02-20', 'Parking', 'Your assigned parking spot is B12 in the underground garage; the gate code is changed monthly.'),
    ('oak-6', 'oak-chat', 'tom@oakresidences.example', [OWNER], '2026-08-01', 'Garden', 'The gardeners will trim the hedges on Saturday morning, sorry in advance for the noise.'),
    # --- Priya at Northwind (client): renewal, payment terms, PO ---
    ('nw-1', 'nw-renewal', 'priya@northwind.example', [OWNER], '2026-01-15', 'Contract renewal', "We'd like to renew the support contract for another 12 months at the current rate."),
    ('nw-2', 'nw-renewal', OWNER, ['priya@northwind.example'], '2026-01-16', 'Re: Contract renewal', 'Happy to renew. Our terms remain net 30 from the invoice date.'),
    ('nw-3', 'nw-renewal', 'priya@northwind.example', [OWNER], '2026-01-18', 'Re: Contract renewal', 'Agreed on net 30. Please reference purchase order PO-88341 on every invoice.'),
    ('nw-4', 'nw-calls', 'priya@northwind.example', [OWNER], '2026-02-01', 'Weekly sync', 'Tuesdays work best for our calls, ideally after 2pm my time (Berlin).'),
    ('nw-5', 'nw-chat', 'priya@northwind.example', [OWNER], '2026-03-01', 'Conference', 'Will you be at the SaaS summit in May? We could grab coffee.'),
    # --- Jordan Park at Northwind (budget) vs Jordan Lee (friend): name collision ---
    ('nwj-1', 'nwj-budget', 'jordan.park@northwind.example', [OWNER], '2026-06-20', 'Q3 budget', 'Finance approved a Q3 budget of 40,000 euros for the integration project.'),
    ('nwj-2', 'nwj-budget', 'jordan.park@northwind.example', [OWNER], '2026-06-22', 'Re: Q3 budget', 'Invoices above 10,000 euros need my sign-off before they are paid.'),
    ('jl-1', 'jl-dinner', 'jordan.lee@friends.example', [OWNER], '2026-09-01', 'Birthday dinner!', "Dinner for my birthday is at Luca's on Saturday at 7pm. Remember I'm allergic to peanuts!"),
    ('jl-2', 'jl-dinner', OWNER, ['jordan.lee@friends.example'], '2026-09-02', 'Re: Birthday dinner!', 'Count me in! Should I bring anything?'),
    ('jl-3', 'jl-dinner', 'jordan.lee@friends.example', [OWNER], '2026-09-02', 'Re: Birthday dinner!', 'Just yourself. Maybe a card, no gifts please.'),
    ('jl-4', 'jl-hike', 'jordan.lee@friends.example', [OWNER], '2026-10-05', 'Hike', 'Up for the Ridge Trail hike next Sunday? Meeting at the north parking lot at 8.'),
    # --- Kim at Ledgerly (accountant) and Alex Kim at Ledgerly (payroll): name collision ---
    ('lg-1', 'lg-tax', 'kim@ledgerly.example', [OWNER], '2026-02-10', 'Tax filing', 'The filing deadline this year is 30 April. Please send your W-2 and all 1099 forms by 15 March.'),
    ('lg-2', 'lg-tax', 'kim@ledgerly.example', [OWNER], '2026-02-11', 'Re: Tax filing', 'Our fee for the personal return is $650, same as last year.'),
    ('lg-3', 'lg-tax', OWNER, ['kim@ledgerly.example'], '2026-03-10', 'Re: Tax filing', 'Attached are my W-2 and two 1099-NEC forms.'),
    ('lg-4', 'lg-payroll', 'alex.kim@ledgerly.example', [OWNER], '2026-05-03', 'Payroll setup', 'Alex here from Ledgerly payroll. Salaries will run on the 25th of each month starting in June.'),
    ('lg-5', 'lg-payroll', 'alex.kim@ledgerly.example', [OWNER], '2026-05-04', 'Re: Payroll setup', 'The payroll service costs $45 per employee per month.'),
    # --- Sofia at TalentBridge (recruiter) ---
    ('tb-1', 'tb-acme', 'sofia@talentbridge.example', [OWNER], '2026-08-10', 'Acme interview', 'Acme would like to interview you on Thursday 20 August at 2pm, video call, about 45 minutes.'),
    ('tb-2', 'tb-acme', 'sofia@talentbridge.example', [OWNER], '2026-08-11', 'Re: Acme interview', 'The salary range for the role is $120k to $140k plus equity.'),
    ('tb-3', 'tb-acme', OWNER, ['sofia@talentbridge.example'], '2026-08-12', 'Re: Acme interview', 'Thursday at 2pm works. Who will be on the panel?'),
    ('tb-4', 'tb-acme', 'sofia@talentbridge.example', [OWNER], '2026-08-13', 'Re: Acme interview', "You'll meet their CTO, Dana Wright, and a senior engineer."),
    ('tb-5', 'tb-other', 'sofia@talentbridge.example', [OWNER], '2026-09-15', 'Another opening', 'A fintech in Lisbon is hiring a staff engineer; remote-friendly. Interested?'),
    # --- GlobeTrek (travel agent): two bookings ---
    ('gt-1', 'gt-lisbon', 'travel@globetrek.example', [OWNER], '2026-04-01', 'Lisbon booking', 'Your flight to Lisbon is confirmed, booking reference QX7K2P, departing 3 May at 08:15 from gate area B.'),
    ('gt-2', 'gt-lisbon', 'travel@globetrek.example', [OWNER], '2026-04-01', 'Lisbon hotel', 'Hotel: Casa do Bairro in Bairro Alto, 4 nights, breakfast included, check-in from 3pm.'),
    ('gt-3', 'gt-porto', 'travel@globetrek.example', [OWNER], '2026-06-10', 'Porto booking', 'Train to Porto booked, reference PT9921, leaving Lisbon Oriente on 7 May at 10:09.'),
    ('gt-4', 'gt-lisbon', OWNER, ['travel@globetrek.example'], '2026-04-02', 'Re: Lisbon booking', 'Can I add a checked bag to the Lisbon flight?'),
    ('gt-5', 'gt-lisbon', 'travel@globetrek.example', [OWNER], '2026-04-03', 'Re: Lisbon booking', 'Done: one 23 kg checked bag added for 35 euros.'),
    # --- Mike at BuildRight (renovation) ---
    ('br-1', 'br-kitchen', 'mike@buildright.example', [OWNER], '2026-05-02', 'Kitchen estimate', 'Our estimate for the kitchen renovation is $18,400 including cabinets and countertops.'),
    ('br-2', 'br-kitchen', 'mike@buildright.example', [OWNER], '2026-05-09', 'Re: Kitchen estimate', 'We can start on 1 July; the work takes about three weeks.'),
    ('br-3', 'br-kitchen', 'mike@buildright.example', [OWNER], '2026-06-20', 'Tiles', 'Bad news: the backsplash tiles you chose are back-ordered until mid-August.'),
    ('br-4', 'br-bathroom', 'mike@buildright.example', [OWNER], '2026-09-01', 'Bathroom quote', 'For the bathroom we estimate $9,800, start possible in October.'),
    # --- Maplewood School ---
    ('ms-1', 'ms-meeting', 'office@maplewood-school.example', [OWNER], '2026-09-20', 'Parent-teacher meeting', 'Parent-teacher meetings are on 12 October; your slot with Ms. Rivera is 17:20.'),
    ('ms-2', 'ms-trip', 'office@maplewood-school.example', [OWNER], '2026-09-25', 'Field trip', 'The science museum field trip is on 3 November; the fee is $25, due by 20 October.'),
    ('ms-3', 'ms-closure', 'office@maplewood-school.example', [OWNER], '2026-10-02', 'School closure', 'The school will be closed on 16 October for staff training.'),
    # --- Iron House gym ---
    ('ih-1', 'ih-pt', 'coach@ironhouse.example', [OWNER], '2026-01-05', 'Personal training', 'The 8-session personal training package is $480. We can do Mondays at 6am.'),
    ('ih-2', 'ih-pt', OWNER, ['coach@ironhouse.example'], '2026-01-06', 'Re: Personal training', "Mondays at 6 are perfect, let's book the package."),
    ('ih-3', 'ih-pt', 'coach@ironhouse.example', [OWNER], '2026-03-01', 'Re: Personal training', 'You have 2 sessions left in your package.'),
    # --- CloudNest (hosting vendor) ---
    ('cn-1', 'cn-migration', 'support@cloudnest.example', [OWNER], '2026-10-20', 'Server migration', 'The server migration is scheduled for Sunday 25 October between 02:00 and 04:00 UTC. Ticket CN-55120.'),
    ('cn-2', 'cn-migration', OWNER, ['support@cloudnest.example'], '2026-10-21', 'Re: Server migration', 'Will the IP address change after the migration?'),
    ('cn-3', 'cn-migration', 'support@cloudnest.example', [OWNER], '2026-10-21', 'Re: Server migration', 'Yes, the new IP will be 203.0.113.42; please update your DNS A record afterwards.'),
    ('cn-4', 'cn-billing', 'support@cloudnest.example', [OWNER], '2026-09-01', 'Billing', 'Your plan renews on 1 December at $29 per month.'),
    # --- Elena at SeedFund (investor): option pool negotiated ---
    ('sf-1', 'sf-terms', 'elena@seedfund.example', [OWNER], '2026-07-14', 'Term sheet', 'Attached is the term sheet: $6M pre-money valuation, $1.5M investment, one board seat.'),
    ('sf-2', 'sf-terms', OWNER, ['elena@seedfund.example'], '2026-07-16', 'Re: Term sheet', 'Thanks Elena. Could the option pool be 10% post-money instead of 15%?'),
    ('sf-3', 'sf-terms', 'elena@seedfund.example', [OWNER], '2026-07-18', 'Re: Term sheet', 'We can meet in the middle at a 12% option pool.'),
    ('sf-4', 'sf-dd', 'elena@seedfund.example', [OWNER], '2026-07-25', 'Due diligence', 'Please upload financials and cap table to the data room by 7 August.'),
    # --- Raj (co-founder) ---
    ('rj-1', 'rj-hiring', 'raj@ourstartup.example', [OWNER], '2026-08-01', 'Hiring plan', 'Proposal: hire two backend engineers in Q4 and one designer in Q1.'),
    ('rj-2', 'rj-hiring', OWNER, ['raj@ourstartup.example'], '2026-08-02', 'Re: Hiring plan', "Agreed on the two engineers, let's revisit the designer after the raise."),
    ('rj-3', 'rj-offsite', 'raj@ourstartup.example', [OWNER], '2026-09-10', 'Offsite', 'I booked a cabin near Lake Tahoe for the founders offsite, 18 to 20 November.'),
    # --- City Bank (mortgage) ---
    ('cb-1', 'cb-rate', 'loans@citybank.example', [OWNER], '2026-07-01', 'Rate lock', 'Your mortgage rate is locked at 4.1% until 20 August. Closing must happen before then.'),
    ('cb-2', 'cb-rate', 'loans@citybank.example', [OWNER], '2026-07-20', 'Documents needed', "To close we still need your last two pay stubs and a homeowner's insurance quote."),
    # --- Safeguard insurance ---
    ('sg-1', 'sg-claim', 'claims@safeguard.example', [OWNER], '2026-03-12', 'Claim SG-7781', 'Your claim SG-7781 for water damage in the kitchen has been approved for $2,300.'),
    ('sg-2', 'sg-claim', 'claims@safeguard.example', [OWNER], '2026-03-19', 'Re: Claim SG-7781', 'The payment will arrive by bank transfer within five business days.'),
    ('sg-3', 'sg-renewal', 'claims@safeguard.example', [OWNER], '2026-10-01', 'Policy renewal', 'Your home policy renews on 1 January; the premium goes up 6% to $1,140 a year.'),
    # --- Nina at Eventful (conference) ---
    ('ev-1', 'ev-venue', 'nina@eventful.example', [OWNER], '2026-08-05', 'Venue options', 'We shortlisted Hall C (300 seats) and the Riverside Loft (180 seats).'),
    ('ev-2', 'ev-venue', OWNER, ['nina@eventful.example'], '2026-08-06', 'Re: Venue options', "Let's take Hall C, we expect about 250 attendees."),
    ('ev-3', 'ev-catering', 'nina@eventful.example', [OWNER], '2026-08-20', 'Catering', 'Catering is planned with 40% vegetarian meals and a vegan option on request.'),
    ('ev-4', 'ev-date', 'nina@eventful.example', [OWNER], '2026-08-21', 'Event date', 'The conference date is confirmed for 4 December, doors open at 8:30.'),
    # --- Luis at Lingua (translation) ---
    ('li-1', 'li-quote', 'luis@lingua.example', [OWNER], '2026-05-15', 'Translation quote', 'For the 12,000-word manual into Spanish we charge 0.09 per word, so $1,080.'),
    ('li-2', 'li-quote', 'luis@lingua.example', [OWNER], '2026-05-16', 'Re: Translation quote', 'Delivery would be on 2 June, with one round of review included.'),
    # --- Ann (neighbour) ---
    ('an-1', 'an-ladder', 'ann.neighbour@friends.example', [OWNER], '2026-10-10', 'Ladder', "Thanks for lending me the ladder! I'll bring it back on Sunday afternoon."),
    ('an-2', 'an-cat', 'ann.neighbour@friends.example', [OWNER], '2026-10-20', 'Cat sitting', "Could you feed Miso while I'm away from 1 to 4 November? Food is in the hallway cupboard."),
    # --- Grace (mentor) ---
    ('gr-1', 'gr-pricing', 'grace@mentors.example', [OWNER], '2026-06-12', 'Pricing advice', 'My advice: anchor the enterprise tier at 3x the team tier and offer annual billing at a 15% discount.'),
    ('gr-2', 'gr-sync', 'grace@mentors.example', [OWNER], '2026-06-12', 'Monthly sync', "Let's keep our mentoring call on the first Friday of every month at 9am."),
    # --- PaperCraft (supplier): two orders ---
    ('pc-1', 'pc-cards', 'orders@papercraft.example', [OWNER], '2026-09-03', 'Order PC-4410', 'Order PC-4410 for 500 business cards has shipped with DHL, tracking 4455 2201 98.'),
    ('pc-2', 'pc-flyers', 'orders@papercraft.example', [OWNER], '2026-09-28', 'Order PC-4471', 'Order PC-4471 for 1,000 flyers is printing and ships next Tuesday.'),
    # --- Dr. Osei (GP) ---
    ('go-1', 'go-labs', 'dr.osei@cityhealth.example', [OWNER], '2026-10-08', 'Lab results', 'Your blood test is normal; total cholesterol 180 mg/dL. No follow-up needed.'),
    ('go-2', 'go-flu', 'dr.osei@cityhealth.example', [OWNER], '2026-10-15', 'Flu shots', 'Flu vaccinations are available from 1 November, walk-ins welcome on weekdays.'),
    # --- Quantum Labs: a second website quote, near-duplicate of Brightweb ---
    ('ql-1', 'ql-quote', 'hello@quantumlabs.example', [OWNER], '2026-06-04', 'Website proposal', 'Our proposal for the website rebuild is $6,900 with a ten-week timeline.'),
    ('ql-2', 'ql-quote', OWNER, ['hello@quantumlabs.example'], '2026-06-07', 'Re: Website proposal', 'Thank you, we decided to go with another agency.'),
    # --- Ben (photographer) ---
    ('bn-1', 'bn-shoot', 'ben@shutterbox.example', [OWNER], '2026-04-14', 'Headshots', 'The team headshot session is on 22 April at our studio; $95 per person, eight people.'),
    ('bn-2', 'bn-shoot', 'ben@shutterbox.example', [OWNER], '2026-04-25', 'Re: Headshots', 'Edited photos are ready; download link expires in 30 days.'),
    # --- Olivia (lawyer) ---
    ('ol-1', 'ol-nda', 'olivia@hartlaw.example', [OWNER], '2026-03-03', 'NDA review', 'I reviewed the NDA; the non-solicitation clause should be limited to 12 months, not 36.'),
    ('ol-2', 'ol-nda', 'olivia@hartlaw.example', [OWNER], '2026-03-04', 'Re: NDA review', 'My fee for the review is $400, billed at the end of the month.'),
    ('ol-3', 'ol-trademark', 'olivia@hartlaw.example', [OWNER], '2026-05-28', 'Trademark', 'The trademark application was filed; serial number 97-554310. Expect 8 months for review.'),
    # --- Felix (freelance developer) ---
    ('fx-1', 'fx-rate', 'felix@devcraft.example', [OWNER], '2026-02-14', 'Availability', "I'm available from March for about 20 hours a week at $85 per hour."),
    ('fx-2', 'fx-rate', OWNER, ['felix@devcraft.example'], '2026-02-15', 'Re: Availability', 'Sounds good. Can you start with the mobile app login flow?'),
    ('fx-3', 'fx-rate', 'felix@devcraft.example', [OWNER], '2026-02-20', 'Re: Availability', "Yes. I'd invoice every two weeks via Wise."),
    ('fx-4', 'fx-vacation', 'felix@devcraft.example', [OWNER], '2026-07-01', 'Vacation', "Heads up: I'm on vacation from 10 to 24 July and offline."),
    # --- Hana (bakery): order changed ---
    ('hb-1', 'hb-cake', 'hana@sweetcrumb.example', [OWNER], '2026-10-02', 'Cake order', 'Your order: a two-tier lemon cake for 30 guests, pick-up 17 October at 11am, $140.'),
    ('hb-2', 'hb-cake', 'hana@sweetcrumb.example', [OWNER], '2026-10-12', 'Re: Cake order', "Change noted: we'll make it raspberry instead of lemon, same price."),
    # --- Victor (office lease) ---
    ('vp-1', 'vp-lease', 'victor@cityoffices.example', [OWNER], '2026-01-20', 'Office lease', 'The office lease runs until 31 March 2027; notice period is three months.'),
    ('vp-2', 'vp-keys', 'victor@cityoffices.example', [OWNER], '2026-02-02', 'Access cards', 'Five new access cards are ready at reception; replacements cost $20 each.'),
    # --- Zoe (journalist) ---
    ('zj-1', 'zj-interview', 'zoe@techdaily.example', [OWNER], '2026-09-18', 'Interview request', 'Could we do a 30-minute interview about your startup for our December issue?'),
    ('zj-2', 'zj-interview', OWNER, ['zoe@techdaily.example'], '2026-09-19', 'Re: Interview request', 'Happy to. Please send the questions in advance; Wednesday afternoons suit me.'),
    ('zj-3', 'zj-interview', 'zoe@techdaily.example', [OWNER], '2026-09-22', 'Re: Interview request', 'Booked for Wednesday 30 September at 3pm; the article goes to print on 15 November.'),
    # --- noise from known people (small talk, out of office, newsletters) ---
    ('nz-1', 'nz-1', 'priya@northwind.example', [OWNER], '2026-04-12', 'Out of office', "I'm out of office until Monday with limited access to email."),
    ('nz-2', 'nz-2', 'mike@buildright.example', [OWNER], '2026-07-15', 'Summer hours', 'Our office closes at 3pm on Fridays during July and August.'),
    ('nz-3', 'nz-3', 'nina@eventful.example', [OWNER], '2026-09-01', 'Newsletter', 'Eventful autumn newsletter: five tips for planning hybrid events.'),
    ('nz-4', 'nz-4', 'kim@ledgerly.example', [OWNER], '2026-06-01', 'Office move', 'Ledgerly has moved to 14 Harbour Street, second floor.'),
    ('nz-5', 'nz-5', 'travel@globetrek.example', [OWNER], '2026-08-15', 'Travel deals', 'Autumn deals: 20% off city breaks to Madrid and Seville.'),
    ('nz-6', 'nz-6', 'elena@seedfund.example', [OWNER], '2026-09-05', 'Portfolio dinner', 'SeedFund portfolio dinner on 12 October, partners welcome.'),
    ('nz-7', 'nz-7', 'raj@ourstartup.example', [OWNER], '2026-09-20', 'Coffee machine', 'The office coffee machine is broken again, I ordered a new one.'),
    ('nz-8', 'nz-8', 'sofia@talentbridge.example', [OWNER], '2026-10-01', 'Market update', 'Hiring market update: salaries for senior engineers rose about 4% this year.'),
    ('nz-9', 'nz-9', 'tom@oakresidences.example', [OWNER], '2026-09-12', 'Recycling', 'Reminder: recycling collection moved to Wednesdays.'),
    ('nz-10', 'nz-10', 'coach@ironhouse.example', [OWNER], '2026-05-01', 'Gym closure', 'The gym is closed on 1 May for the public holiday.'),
    ('nz-11', 'nz-11', 'orders@papercraft.example', [OWNER], '2026-10-10', 'Sale', 'Autumn sale: 15% off all letterpress orders this month.'),
    ('nz-12', 'nz-12', 'alex@brightweb.example', [OWNER], '2026-08-30', 'Case study', "We published a case study about accessible design, thought you'd enjoy it."),
    ('nz-13', 'nz-13', 'dr.chen@riverclinic.example', [OWNER], '2026-06-01', 'Clinic hours', 'From June the clinic opens at 8am on Saturdays.'),
    ('nz-14', 'nz-14', 'grace@mentors.example', [OWNER], '2026-08-08', 'Book', 'Just finished a great book on negotiation, happy to lend it to you.'),
    ('nz-15', 'nz-15', 'jordan.lee@friends.example', [OWNER], '2026-10-20', 'Photos', 'Here are the photos from the hike, the view from the top was unreal.'),
    ('nz-16', 'nz-16', 'luis@lingua.example', [OWNER], '2026-08-01', 'New language', 'We now also offer Portuguese and Japanese translations.'),
    ('nz-17', 'nz-17', 'hana@sweetcrumb.example', [OWNER], '2026-10-25', 'Thank you', 'Thank you for your order, hope everyone enjoyed the cake!'),
    ('nz-18', 'nz-18', 'felix@devcraft.example', [OWNER], '2026-09-01', 'Open source', 'I released a small open-source library for form validation, feedback welcome.'),
    ('nz-19', 'nz-19', 'zoe@techdaily.example', [OWNER], '2026-10-15', 'Photo', 'Could you send a high-resolution photo for the article?'),
    ('nz-20', 'nz-20', 'olivia@hartlaw.example', [OWNER], '2026-09-01', 'Holiday', 'Our office is closed between Christmas and New Year.'),
    ('nz-21', 'nz-21', 'support@cloudnest.example', [OWNER], '2026-07-01', 'Status', 'All systems operational. Scheduled maintenance completed successfully.'),
    ('nz-22', 'nz-22', 'victor@cityoffices.example', [OWNER], '2026-06-15', 'Fire drill', 'Fire drill on Thursday at 11am, please follow the wardens.'),
    ('nz-23', 'nz-23', 'ben@shutterbox.example', [OWNER], '2026-08-01', 'Studio', 'We moved our studio to the Arts Quarter, same phone number.'),
    ('nz-24', 'nz-24', 'claims@safeguard.example', [OWNER], '2026-06-01', 'Tips', 'Five ways to protect your home from storm damage.'),
    ('nz-25', 'nz-25', 'office@maplewood-school.example', [OWNER], '2026-10-20', 'Bake sale', 'The PTA bake sale is on Friday, volunteers welcome.'),
    # --- hard distractors: same people and topic words, wrong or outdated answer ---
    ('hd-1', 'hd-1', 'alex@brightweb.example', [OWNER], '2026-05-20', 'Pricing page', 'Our standard landing page package starts at $1,200; larger sites are quoted individually.'),
    ('hd-2', 'hd-2', 'tom@oakresidences.example', [OWNER], '2025-12-01', 'Annual statement', 'Your 2025 rent statement: 12 payments of $1,250 received, thank you.'),
    ('hd-3', 'hd-3', 'dr.chen@riverclinic.example', [OWNER], '2025-10-02', 'Last visit', 'Thanks for visiting on 2 October last year; your next cleaning is due in spring.'),
    ('hd-4', 'hd-4', 'travel@globetrek.example', [OWNER], '2025-11-12', 'Past trip', 'We hope you enjoyed Madrid! Your previous booking reference was MD4410.'),
    ('hd-5', 'hd-5', 'mike@buildright.example', [OWNER], '2026-04-20', 'Site visit', "I'll measure the kitchen on Friday; the estimate follows next week."),
    ('hd-6', 'hd-6', 'elena@seedfund.example', [OWNER], '2026-06-30', 'Intro call', 'Great talking today. Our typical seed rounds are $1M to $2M at $5M to $8M pre-money.'),
    ('hd-7', 'hd-7', 'kim@ledgerly.example', [OWNER], '2025-04-02', "Last year's return", 'Your 2024 return is filed. The fee was $600.'),
    ('hd-8', 'hd-8', 'coach@ironhouse.example', [OWNER], '2025-11-20', 'Group classes', 'Group classes run Wednesdays at 7pm, $15 per drop-in.'),
    ('hd-9', 'hd-9', 'support@cloudnest.example', [OWNER], '2026-05-05', 'Old ticket', 'Ticket CN-41007 about slow backups is resolved and closed.'),
    ('hd-10', 'hd-10', 'nina@eventful.example', [OWNER], '2026-07-25', 'Venue visit', 'We can tour Hall C and the Riverside Loft next Tuesday.'),
    ('hd-11', 'hd-11', 'orders@papercraft.example', [OWNER], '2026-08-20', 'Proof', 'The proof for your business cards is ready for approval.'),
    ('hd-12', 'hd-12', 'felix@devcraft.example', [OWNER], '2026-01-30', 'Rates 2025', "Last year my rate was $75 per hour; I'm raising it for new projects."),
    ('hd-13', 'hd-13', 'hana@sweetcrumb.example', [OWNER], '2026-09-20', 'Menu', 'Our cakes come in lemon, chocolate and raspberry; two-tier cakes serve up to 40.'),
    ('hd-14', 'hd-14', 'sofia@talentbridge.example', [OWNER], '2026-07-30', 'Profile', 'I shared your profile with Acme and two other companies.'),
    ('hd-15', 'hd-15', 'olivia@hartlaw.example', [OWNER], '2026-02-01', 'Engagement', 'My hourly rate for contract work is $250; reviews under two hours are flat-fee.'),
    ('hd-16', 'hd-16', 'priya@northwind.example', [OWNER], '2025-01-20', "Last year's renewal", 'Renewed for 2025; payment terms were net 45 at the time.'),
    ('hd-17', 'hd-17', 'loans@citybank.example', [OWNER], '2026-06-10', 'Pre-approval', "You're pre-approved; current rates are around 4.3%, which we can lock once you have an offer."),
    ('hd-18', 'hd-18', 'claims@safeguard.example', [OWNER], '2025-08-04', 'Closed claim', 'Claim SG-6120 for a broken window was paid at $410.'),
    ('hd-19', 'hd-19', 'jordan.lee@friends.example', [OWNER], '2026-06-12', 'Picnic', 'Picnic in the park on Saturday at noon, bring a blanket!'),
    ('hd-20', 'hd-20', 'luis@lingua.example', [OWNER], '2026-03-02', 'Rates', 'Our standard rate is 0.11 per word; volume discounts above 10,000 words.'),
]

QUERIES: list[tuple[str, str, str, str, list[str], str]] = [
    ('q01', 'alex@brightweb.example', 'Invoice', 'Before I send the invoice, can you confirm the final amount we agreed for the site?', ['bw-3', 'bw-4'], 'superseded'),
    ('q02', 'alex@brightweb.example', 'Go-live', 'Just double-checking when the new site goes live.', ['bw-5'], 'paraphrase'),
    ('q03', 'alex@brightweb.example', 'Hosting renewal', 'What did we say the annual hosting would cost?', ['bw-6'], 'paraphrase'),
    ('q04', 'maria@lumen-design.example', 'Revisions', 'How many rounds of changes are included in the logo package?', ['ld-1'], 'paraphrase'),
    ('q05', 'maria@lumen-design.example', 'Payment', 'Has invoice LD-2291 been paid yet?', ['ld-3'], 'identifier'),
    ('q06', 'maria@lumen-design.example', 'Concept', 'Which of the concepts did you choose?', ['ld-4'], 'paraphrase'),
    ('q07', 'dr.chen@riverclinic.example', 'Reminder', 'This is a reminder about your upcoming dental visit. Please confirm the time.', ['rc-2'], 'superseded'),
    ('q08', 'dr.chen@riverclinic.example', 'Insurance', 'Can you confirm your policy number for billing?', ['rc-3'], 'identifier'),
    ('q09', 'tom@oakresidences.example', 'Rent', 'Quick reminder about the rent amount from June.', ['oak-1', 'oak-3'], 'paraphrase'),
    ('q10', 'tom@oakresidences.example', 'Heating', 'Was the heating fixed when the engineer came?', ['oak-4'], 'paraphrase'),
    ('q11', 'tom@oakresidences.example', 'Car', 'Which space is yours in the garage?', ['oak-5'], 'paraphrase'),
    ('q12', 'priya@northwind.example', 'Invoice terms', 'Our finance team asks which payment terms and PO we agreed.', ['nw-2', 'nw-3'], 'multi'),
    ('q13', 'priya@northwind.example', 'Next call', 'When is a good day for our next call?', ['nw-4'], 'paraphrase'),
    ('q14', 'jordan.park@northwind.example', 'Budget', 'How much is approved for the integration in Q3?', ['nwj-1'], 'collision'),
    ('q15', 'jordan.park@northwind.example', 'Big invoice', 'We have an invoice for 12,000 euros, who needs to approve it?', ['nwj-2'], 'paraphrase'),
    ('q16', 'jordan.lee@friends.example', 'Saturday', 'What time are we meeting for dinner and where?', ['jl-1'], 'collision'),
    ('q17', 'jordan.lee@friends.example', 'Food', "Remind me what I can't eat? Planning snacks for the hike.", ['jl-1'], 'paraphrase'),
    ('q18', 'kim@ledgerly.example', 'Documents', 'Did you already send your tax documents?', ['lg-3'], 'paraphrase'),
    ('q19', 'kim@ledgerly.example', 'Fee', 'Just confirming the fee for preparing your return.', ['lg-2'], 'paraphrase'),
    ('q20', 'kim@ledgerly.example', 'Deadline', 'When is the filing due again?', ['lg-1'], 'paraphrase'),
    ('q21', 'alex.kim@ledgerly.example', 'Payroll', 'Which day of the month will salaries be paid?', ['lg-4'], 'collision'),
    ('q22', 'alex.kim@ledgerly.example', 'Cost', 'Can you remind me of the per-employee price for payroll?', ['lg-5'], 'collision'),
    ('q23', 'sofia@talentbridge.example', 'Interview prep', 'Any questions before Thursday? Do you remember who you are meeting?', ['tb-4', 'tb-1'], 'multi'),
    ('q24', 'sofia@talentbridge.example', 'Compensation', 'Acme asks for your salary expectations; what range did I share?', ['tb-2'], 'paraphrase'),
    ('q25', 'travel@globetrek.example', 'Check-in', 'Online check-in is open for your Lisbon flight. Do you have your booking code?', ['gt-1'], 'identifier'),
    ('q26', 'travel@globetrek.example', 'Hotel', 'Where are you staying in Lisbon?', ['gt-2'], 'paraphrase'),
    ('q27', 'travel@globetrek.example', 'Train', 'What time is the train to Porto?', ['gt-3'], 'near-duplicate'),
    ('q28', 'travel@globetrek.example', 'Baggage', 'Did you add luggage to your flight?', ['gt-5', 'gt-4'], 'paraphrase'),
    ('q29', 'mike@buildright.example', 'Kitchen start', 'When do we start the kitchen work and how long will it take?', ['br-2'], 'paraphrase'),
    ('q30', 'mike@buildright.example', 'Backsplash', 'Any update on the tiles for the kitchen?', ['br-3'], 'paraphrase'),
    ('q31', 'mike@buildright.example', 'Price', 'What was the kitchen estimate?', ['br-1'], 'near-duplicate'),
    ('q32', 'office@maplewood-school.example', 'Meeting', 'Reminder of your parent-teacher slot.', ['ms-1'], 'paraphrase'),
    ('q33', 'office@maplewood-school.example', 'Museum trip', 'Have you paid for the museum trip?', ['ms-2'], 'paraphrase'),
    ('q34', 'coach@ironhouse.example', 'Package', 'How many sessions do you have left?', ['ih-3'], 'paraphrase'),
    ('q35', 'coach@ironhouse.example', 'Schedule', 'Which day and time do we train?', ['ih-1', 'ih-2'], 'paraphrase'),
    ('q36', 'support@cloudnest.example', 'Ticket CN-55120', 'Following up on ticket CN-55120.', ['cn-1', 'cn-3'], 'identifier'),
    ('q37', 'support@cloudnest.example', 'DNS', 'What is the new IP address after the migration?', ['cn-3'], 'paraphrase'),
    ('q38', 'elena@seedfund.example', 'Option pool', 'Where did we land on the option pool size?', ['sf-3'], 'superseded'),
    ('q39', 'elena@seedfund.example', 'Valuation', 'Remind me of the valuation in the term sheet.', ['sf-1'], 'paraphrase'),
    ('q40', 'elena@seedfund.example', 'Data room', 'When is the deadline for the data room uploads?', ['sf-4'], 'paraphrase'),
    ('q41', 'raj@ourstartup.example', 'Hiring', 'What did we decide about the designer hire?', ['rj-2'], 'paraphrase'),
    ('q42', 'raj@ourstartup.example', 'Offsite', "Where is the founders' retreat and when?", ['rj-3'], 'paraphrase'),
    ('q43', 'loans@citybank.example', 'Closing', 'Reminder: when does your rate lock expire?', ['cb-1'], 'paraphrase'),
    ('q44', 'loans@citybank.example', 'Paperwork', 'Which documents are still missing for closing?', ['cb-2'], 'paraphrase'),
    ('q45', 'claims@safeguard.example', 'Claim', 'Status of claim SG-7781?', ['sg-1', 'sg-2'], 'identifier'),
    ('q46', 'claims@safeguard.example', 'Premium', 'How much will the home insurance cost next year?', ['sg-3'], 'paraphrase'),
    ('q47', 'nina@eventful.example', 'Room', 'Which venue did we book and how many people does it hold?', ['ev-2', 'ev-1'], 'multi'),
    ('q48', 'nina@eventful.example', 'Food', 'Remind me of the vegetarian share for catering.', ['ev-3'], 'paraphrase'),
    ('q49', 'luis@lingua.example', 'Manual', 'When will the Spanish translation be delivered and what does it cost?', ['li-2', 'li-1'], 'multi'),
    ('q50', 'ann.neighbour@friends.example', 'Miso', 'Thanks again for watching Miso, where did I say the food is?', ['an-2'], 'paraphrase'),
    ('q51', 'grace@mentors.example', 'Pricing', 'What did I suggest for the enterprise tier price?', ['gr-1'], 'paraphrase'),
    ('q52', 'orders@papercraft.example', 'Tracking', 'Where is my business card order? I need the tracking number.', ['pc-1'], 'near-duplicate'),
    ('q53', 'dr.osei@cityhealth.example', 'Results', 'What was my cholesterol in the last test?', ['go-1'], 'paraphrase'),
    ('q54', 'ben@shutterbox.example', 'Invoice', 'How many people were in the headshot session and the rate?', ['bn-1'], 'paraphrase'),
    ('q55', 'olivia@hartlaw.example', 'NDA', 'What change did you recommend for the non-solicit?', ['ol-1'], 'paraphrase'),
    ('q56', 'olivia@hartlaw.example', 'Trademark status', "What's the serial number of our trademark filing?", ['ol-3'], 'identifier'),
    ('q57', 'felix@devcraft.example', 'Rate', 'Remind me of my hourly rate and how often I invoice?', ['fx-1', 'fx-3'], 'multi'),
    ('q58', 'hana@sweetcrumb.example', 'Pick-up', 'What flavour is the cake and when do you pick it up?', ['hb-2', 'hb-1'], 'superseded'),
    ('q59', 'victor@cityoffices.example', 'Notice', 'How much notice do we need to give on the lease?', ['vp-1'], 'paraphrase'),
    ('q60', 'zoe@techdaily.example', 'Print date', 'When does the article go to print?', ['zj-3'], 'paraphrase'),
]
# fmt: on
