from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

def expand_shadow_element(element):
  global driver
  shadow_root = driver.execute_script('return arguments[0].shadowRoot.children', element)
  return shadow_root[1]

def guess_word(word):
  global driver
  actions = ActionChains(driver)
  actions.send_keys(word)
  actions.send_keys(Keys.ENTER)
  actions.perform()

def update_evaluations():
  global driver
  global shadow_root_game_app
  global green_letters
  global yellow_letters
  global gray_letters

  game_rows = shadow_root_game_app.find_elements(By.TAG_NAME, 'game-row')

  for game_row in game_rows:
    shadow_root_row = expand_shadow_element(game_row)

    tiles = shadow_root_row.find_elements(By.TAG_NAME, 'game-tile')

    for col in range(5):
      tile = tiles[col]
      evaluation = tile.get_attribute('evaluation')
      letter = tile.get_attribute('letter')

      if evaluation == 'absent':
        if letter not in gray_letters:
          gray_letters.append(letter)

      # We hit a row that has not been evaluated yet
      if evaluation is None:
        return

driver = webdriver.Firefox()
driver.get("https://www.nytimes.com/games/wordle/index.html")

root1 = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "game-app")))
shadow_root_game_app = expand_shadow_element(root1)

root2 = shadow_root_game_app.find_element(By.TAG_NAME, 'game-modal')
shadow_root_game_modal = expand_shadow_element(root2)

close_button = shadow_root_game_modal.find_element(By.TAG_NAME, 'game-icon')

close_button.click()

starting_word = 'slate'

guess_word(starting_word)

word = 'XXXXX'

green_letters = {}
yellow_letters = {}
gray_letters = []

update_evaluations()

print(gray_letters)
