from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

def expand_shadow_element(element):
  shadow_root = driver.execute_script('return arguments[0].shadowRoot.children', element)
  return shadow_root[1]

driver = webdriver.Firefox()
driver.get("https://www.nytimes.com/games/wordle/index.html")

root1 = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "game-app")))
shadow_root1 = expand_shadow_element(root1)

root2 = shadow_root1.find_element(By.TAG_NAME, 'game-modal')
shadow_root2 = expand_shadow_element(root2)

close_button = shadow_root2.find_element(By.TAG_NAME, 'game-icon')

close_button.click()

starting_word = 'slate'

actions = ActionChains(driver)
actions.send_keys(starting_word)
actions.send_keys(Keys.ENTER)
actions.perform()
